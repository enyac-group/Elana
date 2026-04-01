import os
import logging
from functools import partial
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import StaticCache

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

        self.verbose = getattr(args, "verbose", False)

        # Load benchmark data if requested
        self._benchmark_prompts = None
        self._benchmark_idx = 0
        if getattr(args, "benchmark", None):
            self._load_benchmark_data()

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

    # ---------------------- benchmark data ----------------------

    # Supported benchmark datasets: name -> (hf_path, hf_config, split, prompt_builder)
    # prompt_builder(sample) -> str
    BENCHMARK_DATASETS = {
        "humaneval": {
            "hf_path": "openai/openai_humaneval",
            "hf_config": None,
            "split": "test",
            "prompt_builder": lambda s: s["prompt"],
            "description": "Code generation (HumanEval)",
        },
        "gsm8k": {
            "hf_path": "openai/gsm8k",
            "hf_config": "main",
            "split": "test",
            "prompt_builder": lambda s: s["question"],
            "description": "Math reasoning (GSM8K)",
        },
        "triviaqa": {
            "hf_path": "mandarjoshi/trivia_qa",
            "hf_config": "rc.nocontext",
            "split": "validation",
            "prompt_builder": lambda s: s["question"],
            "description": "Knowledge & QA (TriviaQA)",
        },
        "narrativeqa": {
            "hf_path": "deepmind/narrativeqa",
            "hf_config": None,
            "split": "test",
            "prompt_builder": lambda s: (
                s["document"]["summary"] + "\n\nQuestion: " + s["question"]["text"]
            ),
            "description": "Long-context QA (NarrativeQA)",
        },
        "xsum": {
            "hf_path": "EdinburghNLP/xsum",
            "hf_config": None,
            "split": "test",
            "prompt_builder": lambda s: (
                "Summarize the following article:\n\n" + s["document"]
            ),
            "description": "Text summarization (XSum)",
        },
        "ifeval": {
            "hf_path": "google/IFEval",
            "hf_config": None,
            "split": "train",
            "prompt_builder": lambda s: s["prompt"],
            "description": "Instruction following (IFEval)",
        },
    }

    def _load_benchmark_data(self):
        """Load and tokenize benchmark prompts at their natural lengths."""
        from datasets import load_dataset

        benchmark_name = getattr(self.args, "benchmark", "humaneval")
        if benchmark_name not in self.BENCHMARK_DATASETS:
            available = ", ".join(self.BENCHMARK_DATASETS.keys())
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. Available: {available}"
            )

        cfg = self.BENCHMARK_DATASETS[benchmark_name]
        logger.info(
            f"[Rank {self.local_rank}] Loading benchmark dataset: "
            f"{cfg['description']} ({cfg['hf_path']})..."
        )
        dataset = load_dataset(cfg["hf_path"], cfg["hf_config"], split=cfg["split"])

        has_chat_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None
        prompt_builder = cfg["prompt_builder"]

        all_prompts = []
        for sample in dataset:
            text = prompt_builder(sample)
            if has_chat_template:
                messages = [{"role": "user", "content": text}]
                result = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True,
                )
                # apply_chat_template may return a BatchEncoding or a plain tensor
                token_ids = result["input_ids"] if hasattr(result, "keys") else result
            else:
                token_ids = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,
                )["input_ids"]
            # shape: (1, natural_len)
            all_prompts.append(token_ids.to(self.device))

        batch_size = self.args.batch_size
        requested = self.args.repeats * batch_size
        if requested > len(all_prompts):
            logger.warning(
                f"[Rank {self.local_rank}] repeats*batch_size ({self.args.repeats}*{batch_size}={requested}) "
                f"exceeds available prompts ({len(all_prompts)}), "
                f"capping repeats to {len(all_prompts) // batch_size}"
            )
            requested = (len(all_prompts) // batch_size) * batch_size
            self.args.repeats = requested // batch_size
        all_prompts = all_prompts[:requested]

        self._benchmark_prompts = all_prompts
        self._benchmark_idx = 0
        lengths = [p.shape[1] for p in self._benchmark_prompts]
        logger.info(
            f"[Rank {self.local_rank}] Loaded {len(self._benchmark_prompts)} {benchmark_name} prompts "
            f"(token lengths: {min(lengths)}-{max(lengths)}, mean={sum(lengths)/len(lengths):.0f})"
        )

    def _next_benchmark_prompt(self):
        """Return the next benchmark prompt at its natural length (batch_size=1)."""
        prompt = self._benchmark_prompts[self._benchmark_idx]
        self._benchmark_idx = (self._benchmark_idx + 1) % len(self._benchmark_prompts)
        return prompt

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
        if self._benchmark_prompts is not None:
            prompts = [self._next_benchmark_prompt() for _ in range(batch_size)]
            # Pad each to prompt_len and stack
            pad_token_id = self.tokenizer.pad_token_id or 0
            padded = []
            for raw in prompts:
                if raw.shape[1] < prompt_len:
                    raw = torch.nn.functional.pad(raw, (prompt_len - raw.shape[1], 0), value=pad_token_id)
                else:
                    raw = raw[:, -prompt_len:]
                padded.append(raw)
            return torch.cat(padded, dim=0)
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, prompt_len),
            device=self.device,
        )

    def _make_padded_prompt(self, batch_size, prompt_len):
        """Return (prompt, attention_mask) left-padded to prompt_len."""
        if self._benchmark_prompts is not None:
            pad_token_id = self.tokenizer.pad_token_id or 0
            prompts, masks = [], []
            for _ in range(batch_size):
                raw = self._next_benchmark_prompt()  # (1, natural_len)
                natural_len = raw.shape[1]
                pad_len = prompt_len - natural_len
                if pad_len > 0:
                    p = torch.nn.functional.pad(raw, (pad_len, 0), value=pad_token_id)
                    m = torch.cat([
                        torch.zeros(1, pad_len, dtype=torch.long, device=self.device),
                        torch.ones(1, natural_len, dtype=torch.long, device=self.device),
                    ], dim=1)
                else:
                    p = raw[:, -prompt_len:]
                    m = torch.ones(1, prompt_len, dtype=torch.long, device=self.device)
                prompts.append(p)
                masks.append(m)
            return torch.cat(prompts, dim=0), torch.cat(masks, dim=0)
        else:
            prompt = torch.randint(
                low=0, high=self.vocab_size,
                size=(batch_size, prompt_len), device=self.device,
            )
            mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=self.device)
            return prompt, mask

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
            energy_per_unit_joule (float) or 0.0 if energy not enabled.
        """
        if ctx is None:
            return 0.0

        stop_event, power_lists, proc = ctx
        stop_energy_logger_process(proc, stop_event)

        total_joule = 0.0
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
            energy_joule = avg_power * (dur_ms / 1000.0)
            total_joule += energy_joule

        # average over repeats
        energy_per_unit = total_joule / repeats if repeats > 0 else 0.0
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
            for _ in tqdm(range(repeats), desc="Profiling", leave=False):
                fn()
            end.record()
            torch.cuda.synchronize(self.device)
        return start.elapsed_time(end)


    # ---------------------- model size (+ cache) ----------------------

    def profile_size(self, model, batch_size=1, prompt_len=1024, use_GiB=False):

        logger.info(f"[Rank {self.local_rank}] >>> Profiling model size")
        logger.info(f"[Rank {self.local_rank}] Start profiling...")

        device = self.device
        vocab_size = self.vocab_size

        # Dummy prompt for prefilling KV cache
        dummy_prompt = self._make_prompt(batch_size, prompt_len)
        actual_prompt_len = dummy_prompt.shape[1]
        logger.info(f"[Rank {self.local_rank}] Prefilling {actual_prompt_len} tokens to KV cache...")
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
        logger.info(f'[Rank {self.local_rank}] cache size: {cache_size_gb:.3f} GB (batch size {batch_size}, prompt length {actual_prompt_len})')
        # model total size and detailed layer type breakdown
        model_size_gb = (param_size + buffer_size) / GB
        logger.info(f'[Rank {self.local_rank}] model size: {model_size_gb:.3f} GB')
        return model_size_gb, cache_size_gb


    # ---------------------- TTFT ----------------------

    def profile_ttft(self, batch_size=1, prompt_len=1024,
                     repeats=100, torch_profile=False, torch_profile_dir=""):
        logger.info(f"[Rank {self.local_rank}] >>> Profiling TTFT (prefilling stage) for {repeats} times")

        use_benchmark = self._benchmark_prompts is not None

        if use_benchmark:
            logger.info(
                f"[Rank {self.local_rank}] Testing with HumanEval prompts at natural lengths (batch_size=1)"
            )
        else:
            logger.info(f"[Rank {self.local_rank}] Testing (batch_size, prompt_len): ({batch_size}, {prompt_len})")

        logger.info(f"[Rank {self.local_rank}] Warmup...")
        with torch.no_grad():
            for _ in range(5):
                p = self._make_prompt(batch_size, prompt_len)
                _ = self.model(
                    p,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
        torch.cuda.synchronize()

        logger.info(f"[Rank {self.local_rank}] Start profiling...")
        energy_ctx = self._maybe_start_energy_logger()

        def _run_once():
            p = self._make_prompt(batch_size, prompt_len)
            _ = self.model(
                p,
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
            outfile_prefix = f"ttft_prompt_len_{'humaneval' if use_benchmark else prompt_len}"

            def _inner(prof):
                for _ in range(5):
                    with record_function("## forward ##"):
                        p = self._make_prompt(batch_size, prompt_len)
                        _ = self.model(
                            p,
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

        # Prompt for prefilling KV cache
        dummy_prompt = self._make_prompt(batch_size, prompt_len)

        logger.info(f"[Rank {self.local_rank}] Prefilling {dummy_prompt.shape[1]} tokens to KV cache...")
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

        # Variables for cache_graph path (StaticCache)
        static_cache = None
        prefill_positions = None

        # cache the graph for generation using StaticCache
        if cache_graph:
            torch.cuda.set_device(self.device)

            static_cache = StaticCache(
                config=self.model.config,
                batch_size=batch_size,
                max_cache_len=prompt_len + gen_len,
                device=device,
                dtype=self.dtype,
            )
            prefill_positions = torch.arange(prompt_len, device=device)

            # Prefill with dummy prompt to populate static_cache
            # Use random tokens (not _make_padded_prompt) to guarantee correct batch_size,
            # since benchmark prompts always return batch=1.
            prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
            mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
            with torch.no_grad():
                self.model(
                    prompt,
                    attention_mask=mask,
                    past_key_values=static_cache,
                    cache_position=prefill_positions,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )

            input_token = torch.randint(
                low=0, high=vocab_size, size=(batch_size, 1), device=device
            )

            # Pre-allocate a static attention mask for CUDA graph capture.
            # Shape: (batch_size, prompt_len + gen_len) — content updated via copy_() before replay.
            static_attn_mask = torch.ones(batch_size, prompt_len + gen_len, dtype=torch.long, device=device)

            # Warmup decode steps on a side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = self.model(
                            input_token,
                            attention_mask=static_attn_mask,
                            past_key_values=static_cache,
                            cache_position=cache_position,
                            use_cache=True,
                            output_hidden_states=False,
                            output_attentions=False,
                        )
            torch.cuda.current_stream().wait_stream(s)

            # Capture CUDA graph for decode step
            with torch.no_grad():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.model(
                        input_token,
                        attention_mask=static_attn_mask,
                        past_key_values=static_cache,
                        cache_position=cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )

            def generate(new_input_token, new_attn_mask):
                input_token.copy_(new_input_token)
                static_attn_mask.copy_(new_attn_mask)
                graph.replay()
                return out
        else:
            def generate(new_input_token, new_past_key_values, attention_mask=None):
                out = self.model(
                    new_input_token,
                    attention_mask=attention_mask,
                    past_key_values=new_past_key_values,
                    cache_params=new_past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                return out

        def run_once(batch_size, prompt_len, gen_len, verbose=False):
            if static_cache is not None:
                # ---- cache_graph path (StaticCache) ----
                static_cache.reset()
                prompt, mask = self._make_padded_prompt(batch_size, prompt_len)
                actual_prompt_len = prompt.shape[1]
                sequences = [prompt]

                # Prefill with static_cache
                outputs = self.model(
                    prompt,
                    attention_mask=mask,
                    past_key_values=static_cache,
                    cache_position=prefill_positions,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )

                sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                sequences.append(sampled_tokens)

                # Build full-length attention mask for decode steps.
                # Start from the prefill mask, extend to max_cache_len, and grow
                # the valid region by 1 at each decode step.
                max_cache_len = prompt_len + gen_len
                decode_mask = torch.zeros(batch_size, max_cache_len, dtype=torch.long, device=device)
                decode_mask[:, :actual_prompt_len] = mask
                decode_mask[:, actual_prompt_len] = 1  # first generated token

                # Generation loop with CUDA graph replay
                eos_token_id = self.tokenizer.eos_token_id
                for i in range(gen_len - 1):
                    if eos_token_id is not None and (sampled_tokens == eos_token_id).all():
                        break
                    cache_position.fill_(actual_prompt_len + i)
                    decode_mask[:, actual_prompt_len + i] = 1
                    outputs = generate(sequences[-1], decode_mask)
                    sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    sequences.append(sampled_tokens)

                actual_gen_len = len(sequences) - 1
                run_stats.append((actual_prompt_len, actual_gen_len))

                if verbose and self._benchmark_prompts is not None:
                    generated_tokens = torch.cat(sequences[1:], dim=-1)
                    for b in range(prompt.shape[0]):
                        input_text = self.tokenizer.decode(prompt[b], skip_special_tokens=True)
                        output_text = self.tokenizer.decode(generated_tokens[b], skip_special_tokens=True)
                        logger.info(f"[Rank {self.local_rank}] [Batch {b}] [INPUT]\n{input_text}")
                        logger.info(f"[Rank {self.local_rank}] [Batch {b}] [OUTPUT]\n{output_text}")
            else:
                # ---- non-graph path (DynamicCache) ----
                prompt, mask = self._make_padded_prompt(batch_size, prompt_len)
                actual_prompt_len = prompt.shape[1]
                sequences = [prompt]

                # prefilling
                outputs = self.model(
                    sequences[-1],
                    attention_mask=mask,
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

                # generation — extend mask by 1 at each step
                eos_token_id = self.tokenizer.eos_token_id
                current_past_key_values = past_key_values
                for i in range(gen_len - 1):
                    if eos_token_id is not None and (sampled_tokens == eos_token_id).all():
                        break
                    mask = torch.cat([mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
                    cache_position.fill_(actual_prompt_len + i)
                    outputs = generate(sequences[-1], current_past_key_values, attention_mask=mask)
                    if hasattr(outputs, "past_key_values"):
                        current_past_key_values = outputs.past_key_values
                    else:
                        current_past_key_values = outputs.cache_params
                    sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    sequences.append(sampled_tokens)

                actual_gen_len = len(sequences) - 1
                run_stats.append((actual_prompt_len, actual_gen_len))

                if verbose and self._benchmark_prompts is not None:
                    generated_tokens = torch.cat(sequences[1:], dim=-1)
                    for b in range(prompt.shape[0]):
                        input_text = self.tokenizer.decode(prompt[b], skip_special_tokens=True)
                        output_text = self.tokenizer.decode(generated_tokens[b], skip_special_tokens=True)
                        logger.info(f"[Rank {self.local_rank}] [Batch {b}] [INPUT]\n{input_text}")
                        logger.info(f"[Rank {self.local_rank}] [Batch {b}] [OUTPUT]\n{output_text}")

        run_stats = []

        logger.info(f"[Rank {self.local_rank}] Warmup...")
        with torch.no_grad():
            for _ in range(5):
                run_once(batch_size, prompt_len, gen_len, verbose=False)

        run_stats.clear()

        logger.info(f"[Rank {self.local_rank}] Start profiling...")
        energy_ctx = self._maybe_start_energy_logger()

        def _run():
            run_once(batch_size, prompt_len, gen_len, verbose=self.verbose)

        dur = self._time_repeated(repeats, _run)
        avg_ms = dur / repeats

        if run_stats:
            avg_prompt = sum(s[0] for s in run_stats) / len(run_stats)
            avg_gen = sum(s[1] for s in run_stats) / len(run_stats)
            logger.info(
                f"[Rank {self.local_rank}] Finished, latency: {avg_ms:.2f} ms (cache_graph={cache_graph}), "
                f"avg prompt_len: {avg_prompt:.0f}, avg gen_len: {avg_gen:.0f}"
            )
        else:
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
                    run_once(batch_size, prompt_len, gen_len, verbose=False)
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
        if self._benchmark_prompts is not None:
            num_prompts = len(self._benchmark_prompts)
            lengths = [p.shape[1] for p in self._benchmark_prompts]
            args.prompt_len = max(lengths)
            logger.info(
                f"[Rank {self.local_rank}] Using {args.benchmark} benchmark prompts "
                f"({num_prompts} problems, repeats={args.repeats}, batch_size={micro_batch_size}, "
                f"prompt_len auto-set to {args.prompt_len} (max of {min(lengths)}-{max(lengths)}))"
            )
        else:
            logger.info(f"[Rank {self.local_rank}] Using random token inputs")
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

