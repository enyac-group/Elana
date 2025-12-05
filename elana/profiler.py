import os
import logging
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.profiler import record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from .size import dynamic_cache_nbytes
from .trace_handler import trace_handler
from .energy import launch_energy_logger_process
from .energy import stop_energy_logger_process
from .logger_utils import set_logger

logger = logging.getLogger(__file__)


class ElanaProfiler:
    """
    ElanaProfiler:
      - Loads a causal LM + tokenizer from HF (local dir or hub).
      - Profiles size, TTFT, TPOT, TTLT with optional energy and torch profiler.
    """

    def __init__(self, args, dtype=torch.bfloat16, device_map="auto"):
        self.args = args
        self.dtype = dtype

        # Get per-process device from torchrun (LOCAL_RANK will be set)
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # For CUDA graphs, we must be single-GPU, so override device_map
        if getattr(args, "cache_graph", False):
            logger.warning(f"[Rank {self.local_rank}] CUDA graphs must be single-GPU, overriding device_map to None")
            device_map = None

        logger.info(
            f"[Rank {self.local_rank}] ElanaProfiler initialized with dtype={self.dtype}, "
            f"device_map={device_map}, device={self.device}"
        )
        self.device_map = device_map
        self.model_name = getattr(args, "model_name", None)
        if self.model_name is None:
            self.model_name = str(args.model_repo).split("/")[-1]

        logger.info(f"[Rank {self.local_rank}] Loading model from {args.model_repo}, model name: {self.model_name}")

        # Build model & tokenizer
        self.model, self.tokenizer = self._hf_build_model_and_tokenizer()
        self.model.eval()

        # Convenience handle
        self.vocab_size = self.model.config.vocab_size

    # ---------------------- model loading ----------------------

    def _hf_build_model_and_tokenizer(self):
        """
        Standard HF loading path for inference / profiling.

        With torchrun:
        - Each rank gets its own full model replica on a single GPU.
        - No DDP wrapping, so all HF methods (generate, prepare_inputs_for_generation, etc.)
            remain available on self.model.
        """

        # --- Tokenizer ---
        logger.info(f"[Rank {self.local_rank}] Loading tokenizer from {self.args.model_repo}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_repo,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Device / rank ---
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device)
        else:
            device = self.device  # whatever you set (e.g., torch.device("cuda:0"))

        logger.info(
            f"[Rank {self.local_rank}] Loading model from {self.args.model_repo} "
            f"with dtype={self.dtype} on {device}..."
        )

        # --- Model ---
        if is_distributed or self.device_map is None:
            # Single GPU per process
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_repo,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
        else:
            # Single-process multi-GPU sharding
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_repo,
                dtype=self.dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        return model, tokenizer

    # ---------------------- common helpers ----------------------

    def _run_torch_profiler(
        self,
        outfile_prefix: str,
        torch_profile_dir: str,
        use_gzip: bool,
        inner_loop_fn,
        warn_msg: str | None = None,
    ):
        """
        Wrap torch.profiler.profile + common settings.

        inner_loop_fn(prof) should:
          - run the target workload several times
          - call prof.step() each iteration
        """
        logger.info(f"[Rank {self.local_rank}] Run torch profiler...")
        if warn_msg:
            logger.warning(f"[Rank {self.local_rank}] {warn_msg}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler,
                dir_name=torch_profile_dir,
                use_gzip=use_gzip,
                file_prefix=outfile_prefix,
                device="cuda:0",
            ),
        ) as prof:
            with torch.no_grad():
                inner_loop_fn(prof)

    def _make_prompt(self, batch_size, prompt_len):
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, prompt_len),
            device=self.device,
        )

    def _maybe_start_energy_logger(self):
        if not getattr(self.args, "energy", False):
            return None
        logger.info(f"[Rank {self.local_rank}] Launch energy logger process...")
        stop_event, power_list, proc = launch_energy_logger_process()
        return (stop_event, power_list, proc)

    def _maybe_finish_energy_logger(self, ctx, dur_ms, repeats, unit, truncate_head_tail=False):
        """
        Stop energy logger and report average power + per-unit energy.
        - dur_ms: total duration in ms
        - repeats: number of measured runs
        - unit: "prompt", "token", or "request"

        Returns:
            energy_per_unit_joules (float) or 0.0 if energy not enabled.
        """
        if ctx is None:
            return 0.0

        stop_event, power_lists, proc = ctx
        stop_energy_logger_process(proc, stop_event)

        total_joules = 0.0
        for gpu_index, gpu_power in enumerate(power_lists):
            watts = list(gpu_power)

            if truncate_head_tail and len(watts) > 10:
                n_trunc = len(watts) // 10
                watts = watts[n_trunc:-n_trunc]

            avg_power = sum(watts) / len(watts) if watts else 0.0
            logger.info(
                f"[Rank {self.local_rank}] Collected power {len(watts)} samples, "
                f"avg power = {avg_power:.2f} W"
            )
            energy_joules = avg_power * (dur_ms / 1000.0)
            total_joules += energy_joules

        # average over repeats
        energy_per_unit = total_joules / repeats if repeats > 0 else 0.0
        logger.info(f"[Rank {self.local_rank}] Total energy for all GPUs: {energy_per_unit:.2f} J / {unit}")
        return energy_per_unit

    def _time_repeated(self, repeats, fn):
        """
        Time `fn()` repeated `repeats` times using CUDA events.
        Returns total duration in milliseconds.
        """
        with torch.no_grad():
            torch.cuda.set_device(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            torch.cuda.synchronize(self.device)
        return start.elapsed_time(end)


    # ---------------------- model size (+ cache) ----------------------

    def profile_size(self, model, batch_size=1, prompt_len=1024, use_GiB=False):

        logger.info(">>> Profiling model size")
        logger.info("Start profiling...")

        device = self.device
        vocab_size = self.vocab_size

        # Dummy prompt for prefilling KV cache
        dummy_prompt = torch.randint(
            low=0, high=vocab_size, size=(batch_size, prompt_len), device=device
        )
        logger.info("Prefilling KV cache...")
        if hasattr(model, "prepare_inputs_for_generation"):
            # Nemotron-H and Mamba2
            with torch.no_grad():
                model_inputs = self.model.prepare_inputs_for_generation(dummy_prompt, use_cache=True, cache_params= None)
                if "past_key_values" in model_inputs: # Nemotron-H
                    past_key_values = model_inputs["past_key_values"]
                elif "cache_params" in model_inputs: # mamba2
                    past_key_values = model_inputs["cache_params"]
                else:
                    logger.error(f"Model inputs keys: {model_inputs.keys()}")
                    raise ValueError("Model does not have past_key_values or cache_params after prepare_inputs_for_generation")
        # HOTFIX (HY): Llama models will not get cache from prepare_inputs_for_generation
        # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py#L546
        if past_key_values is None:
            with torch.no_grad():
                outputs = self.model(
                    dummy_prompt,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                if hasattr(outputs, "past_key_values"):
                    past_key_values = outputs.past_key_values
                else:
                    raise ValueError("past_key_values is missing in the model output")

        # get model total size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # model conv/ssm/kv caches
        GB = 1000**3 # This is the SI (base-10) definition used by most storage manufacturers.
        if use_GiB:
            GB = 1024**3 # Binary (used by Linux/OS), aka GiB
        # print(past_key_values)
        cache_size = dynamic_cache_nbytes(past_key_values)
        cache_size_gb = cache_size / GB
        logger.info(f'[Rank {self.local_rank}] cache size: {cache_size_gb:.3f} GB (batch size {batch_size}, prompt length {prompt_len})')
        # model total size and detailed layer type breakdown
        model_size_gb = (param_size + buffer_size) / GB
        logger.info(f'[Rank {self.local_rank}] model size: {model_size_gb:.3f} GB')
        return model_size_gb, cache_size_gb


    # ---------------------- TTFT ----------------------

    def profile_ttft(self, batch_size=1, prompt_len=1024,
                     repeats=100, torch_profile=False, torch_profile_dir=""):
        logger.info(f"[Rank {self.local_rank}] >>> Profiling TTFT (prefilling stage) for {repeats} times")

        prompt = self._make_prompt(batch_size, prompt_len)

        logger.info(f"[Rank {self.local_rank}] Testing (batch_size, prompt_len): ({batch_size}, {prompt_len})")
        logger.info(f"[Rank {self.local_rank}] Warmup...")
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(
                    prompt,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
        torch.cuda.synchronize()

        logger.info(f"[Rank {self.local_rank}] Start profiling...")
        energy_ctx = self._maybe_start_energy_logger()

        def _run_once():
            _ = self.model(
                prompt,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )

        dur = self._time_repeated(repeats, _run_once)
        avg_ms = dur / repeats
        logger.info(f"[Rank {self.local_rank}] Finished, latency: {avg_ms:.2f} milliseconds")

        # TTFT: energy per prompt, truncate head/tail as in original code
        energy_prompt = self._maybe_finish_energy_logger(
            energy_ctx, dur, repeats, unit="prompt", truncate_head_tail=True
        )

        if torch_profile:
            outfile_prefix = f"ttft_prompt_len_{prompt_len}"

            def _inner(prof):
                for _ in range(5):
                    with record_function("## forward ##"):
                        _ = self.model(
                            prompt,
                            use_cache=True,
                            output_hidden_states=False,
                            output_attentions=False,
                        )
                    prof.step()

            self._run_torch_profiler(
                outfile_prefix=outfile_prefix,
                torch_profile_dir=torch_profile_dir,
                use_gzip=True,           # TTFT: you were using gzip=True
                inner_loop_fn=_inner,
                warn_msg=None,
            )
        
        return avg_ms, energy_prompt

    # ---------------------- TPOT ----------------------

    def profile_tpot(self, batch_size=1, prompt_len=1024,
                     repeats=100, cache_graph=False,
                     torch_profile=False, torch_profile_dir=""):
        logger.info(f"[Rank {self.local_rank}] >>> Profiling TPOT (generation stage) for {repeats} times, cache_graph: {cache_graph}")

        device = self.device
        vocab_size = self.vocab_size

        # Dummy prompt for prefilling KV cache
        dummy_prompt = torch.randint(
            low=0, high=vocab_size, size=(batch_size, prompt_len), device=device
        )

        logger.info("Prefilling KV cache...")
        with torch.no_grad():
            outputs = self.model(
                dummy_prompt,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            if hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
            else:
                # Nemotron-H and Mamba2
                past_key_values = outputs.cache_params

        # Single-token input for generation
        input_token = torch.randint(low=0, high=vocab_size, size=(batch_size, 1), device=device)
        cache_position = torch.arange(1, device=device)

        # Warmup
        logger.info(f"[Rank {self.local_rank}] Warmup...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(s):
                for _ in range(5):
                    _ = self.model(
                        input_token,
                        past_key_values=past_key_values,
                        cache_params=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )
        torch.cuda.current_stream().wait_stream(s)

        if cache_graph:
            torch.cuda.set_device(self.device)  # NEW
            with torch.no_grad():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.model(
                        input_token,
                        past_key_values=past_key_values,
                        cache_params=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )

            def generate(new_input_token, new_past_key_values):
                input_token.copy_(new_input_token)
                # Simplified CUDA graph usage (past_key_values static at capture time)
                graph.replay()
                return out
        else:
            def generate(new_input_token, new_past_key_values):
                out = self.model(
                    new_input_token,
                    past_key_values=new_past_key_values,
                    cache_params=new_past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                return out

        logger.info(f"[Rank {self.local_rank}] Start profiling...")
        energy_ctx = self._maybe_start_energy_logger()

        new_input_token = torch.randint(
            low=0, high=vocab_size, size=(batch_size, 1), device=device
        )

        def _run_once():
            generate(new_input_token, past_key_values)

        dur = self._time_repeated(repeats, _run_once)
        avg_ms = dur / repeats
        logger.info(
            f"[Rank {self.local_rank}] Finished, latency: {avg_ms:.2f} milliseconds (cache_graph={cache_graph})"
        )
        # TPOT: energy per token
        energy_per_token = self._maybe_finish_energy_logger(energy_ctx, dur, repeats, unit="token")

        if torch_profile:
            outfile_prefix = "tpot"

            def _inner(prof):
                for _ in range(5):
                    generate(new_input_token, past_key_values)
                    prof.step()

            self._run_torch_profiler(
                outfile_prefix=outfile_prefix,
                torch_profile_dir=torch_profile_dir,
                use_gzip=False,          # TPOT: you used gzip=False
                inner_loop_fn=_inner,
                warn_msg=None,
            )
        
        return avg_ms, energy_per_token

    # ---------------------- TTLT ----------------------

    def profile_ttlt(self, batch_size=1, prompt_len=1024, gen_len=128,
                     repeats=100, cache_graph=False,
                     torch_profile=False, torch_profile_dir=""):
        logger.info(
            f"[Rank {self.local_rank}] >>> Profiling TTLT (prefilling + generation) for {repeats} times, cache_graph: {cache_graph}"
        )
        logger.info(
            f"[Rank {self.local_rank}] batch_size: {batch_size}, prompt_len: {prompt_len}, gen_len:{gen_len}"
        )

        device = self.device
        vocab_size = self.vocab_size
        cache_position = torch.arange(1, device=device)

        # cache the graph for generation
        if cache_graph:
            torch.cuda.set_device(self.device)  # NEW
            dummy_prompt = torch.randint(
                low=0, high=vocab_size, size=(batch_size, prompt_len), device=device
            )
            with torch.no_grad():
                outputs = self.model(
                    dummy_prompt,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                if hasattr(outputs, "past_key_values"):
                    past_key_values = outputs.past_key_values
                else:
                    past_key_values = outputs.cache_params

            input_token = torch.randint(
                low=0, high=vocab_size, size=(batch_size, 1), device=device
            )
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = self.model(
                            input_token,
                            past_key_values=past_key_values,
                            cache_params=past_key_values,
                            cache_position=cache_position,
                            use_cache=True,
                            output_hidden_states=False,
                            output_attentions=False,
                        )
            torch.cuda.current_stream().wait_stream(s)

            with torch.no_grad():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.model(
                        input_token,
                        past_key_values=past_key_values,
                        cache_params=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )

            def generate(new_input_token, new_past_key_values):
                input_token.copy_(new_input_token)
                graph.replay()
                return out
        else:
            def generate(new_input_token, new_past_key_values):
                out = self.model(
                    new_input_token,
                    past_key_values=new_past_key_values,
                    cache_params=new_past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                return out

        def run_once(batch_size, prompt_len, gen_len):
            prompt = torch.randint(
                low=0, high=vocab_size, size=(batch_size, prompt_len), device=device
            )
            sequences = [prompt]

            # prefilling
            outputs = self.model(
                sequences[-1],
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            if hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
            else:
                past_key_values = outputs.cache_params

            sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            sequences.append(sampled_tokens)

            # generation
            current_past_key_values = past_key_values
            for _ in range(gen_len - 1):  # one token already generated
                outputs = generate(sequences[-1], current_past_key_values)
                if hasattr(outputs, "past_key_values"):
                    current_past_key_values = outputs.past_key_values
                else:
                    current_past_key_values = outputs.cache_params
                sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                sequences.append(sampled_tokens)

        logger.info(f"[Rank {self.local_rank}] Warmup...")
        with torch.no_grad():
            for _ in range(5):
                run_once(batch_size, prompt_len, gen_len)

        logger.info(f"[Rank {self.local_rank}] Start profiling...")
        energy_ctx = self._maybe_start_energy_logger()

        def _run():
            run_once(batch_size, prompt_len, gen_len)

        dur = self._time_repeated(repeats, _run)
        avg_ms = dur / repeats
        logger.info(
            f"[Rank {self.local_rank}] Finished, latency: {avg_ms:.2f} milliseconds (cache_graph={cache_graph})"
        )
        # TTLT: energy per request
        energy_request = self._maybe_finish_energy_logger(energy_ctx, dur, repeats, unit="request")

        if torch_profile:
            outfile_prefix = (
                f"ttlt_prompt_len_{prompt_len}_gen_len_{gen_len}_cache_graph_{cache_graph}"
            )

            def _inner(prof):
                for _ in range(5):
                    run_once(batch_size, prompt_len, gen_len)
                    prof.step()

            self._run_torch_profiler(
                outfile_prefix=outfile_prefix,
                torch_profile_dir=torch_profile_dir,
                use_gzip=False,
                inner_loop_fn=_inner,
                warn_msg="Profile ttlt with torch profiler is very slow...",
            )

        return avg_ms, energy_request

    # ---------------------- main entry (was main(args)) ----------------------

    def run(self):
        args = self.args
        model = self.model
        model_name = self.model_name
        micro_batch_size = args.batch_size // getattr(args, "world_size", 1)
        logger.info(f"[Rank {self.local_rank}] Using micro-batch size: {micro_batch_size}")
        metrics = {}

        # ---- size profiling ----
        if args.size:
            if args.energy:
                logger.warning(f"[Rank {self.local_rank}] Model size profiling does not support energy measurement, ignore --energy")
            if args.cache_graph:
                logger.warning(f"[Rank {self.local_rank}] Model size profiling does not support cache_graph mode, ignore --cache_graph")
            model_size_gb, cache_size_gb = self.profile_size(model, micro_batch_size, args.prompt_len)
            metrics["model_size_gb"] = model_size_gb
            metrics["cache_size_gb"] = cache_size_gb

        # ---- TTFT ----
        if args.ttft:
            if args.cache_graph:
                logger.warning(f"[Rank {self.local_rank}] TTFT does not support cache_graph mode, ignore --cache_graph")
            ttft_latency_ms, ttft_energy_j = self.profile_ttft(
                batch_size=micro_batch_size,
                prompt_len=args.prompt_len,
                repeats=args.repeats,
                torch_profile=args.torch_profile,
                torch_profile_dir=f"torch_profile/{model_name}",
            )
            metrics["ttft_latency_ms"] = ttft_latency_ms
            if args.energy:
                metrics["ttft_energy_j_per_prompt"] = ttft_energy_j

        # ---- TPOT ----
        if args.tpot:
            if args.gen_len > 1:
                logger.warning(f"[Rank {self.local_rank}] TPOT only tests the latency with the given prompt length, ignore --gen_len")
            tpot_latency_ms, tpot_energy_j = self.profile_tpot(
                batch_size=micro_batch_size,
                prompt_len=args.prompt_len,
                repeats=args.repeats,
                cache_graph=args.cache_graph,
                torch_profile=args.torch_profile,
                torch_profile_dir=f"torch_profile/{model_name}",
            )
            metrics["tpot_latency_ms"] = tpot_latency_ms
            if args.energy:
                metrics["tpot_energy_j_per_token"] = tpot_energy_j

        # ---- TTLT ----
        if args.ttlt:
            ttlt_latency_ms, ttlt_energy_j = self.profile_ttlt(
                batch_size=micro_batch_size,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                repeats=args.repeats,
                cache_graph=args.cache_graph,
                torch_profile=args.torch_profile,
                torch_profile_dir=f"torch_profile/{model_name}",
            )
            metrics["ttlt_latency_ms"] = ttlt_latency_ms
            if args.energy:
                metrics["ttlt_energy_j_per_request"] = ttlt_energy_j

        if not args.size and not args.ttft and not args.tpot and not args.ttlt:
            logger.warning(
                f"[Rank {self.local_rank}] No profiling task to run with, try `--ttft`, `--tpot`, `--ttlt`, `--size`?"
            )

        return metrics

