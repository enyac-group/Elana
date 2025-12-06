# ddp_main.py
import os
import argparse
import logging

import torch
import torch.distributed as dist

from elana.profiler import ElanaProfiler  # <-- adjust to your module path
from elana.logger_utils import set_logger  # reuse your logger setup if you want

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ElanaProfiler DDP launcher")

    # Core Elana args (match what your current CLI expects)
    parser.add_argument("model_repo", type=str,
                        help="HF repo or local path, e.g. meta-llama/Llama-2-7b-hf")

    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)

    parser.add_argument("--size", action="store_true")
    parser.add_argument("--ttft", action="store_true")
    parser.add_argument("--tpot", action="store_true")
    parser.add_argument("--ttlt", action="store_true")

    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--cache_graph", action="store_true")
    parser.add_argument("--torch_profile", action="store_true")

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map; will be overridden to None when --cache_graph is used.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def setup_logging(args, rank):
    level = getattr(logging, args.log_level.upper())
    root_logger = logging.getLogger()
    set_logger(root_logger, level)

    # Optional: only show INFO+ from rank 0, suppress others a bit
    if rank != 0:
        # Reduce noise from non-root ranks
        logging.getLogger().setLevel(max(level, logging.ERROR))
    if rank == 0:
        logger.highlight(f"Logging info will only show on rank {rank}")


def setup_distributed():
    """
    Initialize torch.distributed using env vars from torchrun.
    """
    if dist.is_initialized():
        return

    # torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    args.world_size = world_size
    args.local_rank = local_rank
    assert args.batch_size % world_size == 0, \
        f"Batch size {args.batch_size} must be divisible by world size {world_size}"
    assert args.batch_size >= world_size, \
        f"Batch size {args.batch_size} must be >= world size {world_size}"

    setup_logging(args, rank)
    logger.highlight(f"Launching distributed ElanaProfiler with {world_size} ranks.")

    # Construct profiler (it will read LOCAL_RANK and bind itself to that GPU)
    profiler = ElanaProfiler(
        args,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )

    local_metrics = profiler.run()  # dict with tpot_latency_ms, tpot_energy_j_per_token

    # sync
    dist.barrier()
    if rank == 0:
        logger.highlight(f"All ranks finished Elana profiling.")
        logger.highlight(f"Summarizing the energy, latency, and size from {world_size} ranks.")


    # ---- Global reduction (example for TPOT only) ----
    for key in local_metrics:
        value = local_metrics.get(key, 0.0)
        tensor = torch.tensor(
            [value],
            device=f"cuda:{local_rank}",
            dtype=torch.float32,
        )
        # Sum across ranks
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if "energy" in key:
            if rank == 0:
                logger.info(f"[GLOBAL] {key}: {tensor.item():.2f} Joule in total")  # sum energy
        elif "latency" in key:
            tensor /= world_size  # average latency
            if rank == 0:
                logger.info(f"[GLOBAL] {key}: {tensor.item():.2f} ms average")
        elif "model_size" in key:
            if rank == 0:
                logger.info(f"[GLOBAL] {key}: {value:.2f} GB at rank {rank}") # we only use the value from rank 0
        elif "cache_size" in key:
            if rank == 0:
                logger.info(f"[GLOBAL] {key}: {value:.2f} GB at rank {rank}, {tensor.item():.2f} GB in total") # we only use the value from rank 0
        else:
            raise ValueError(f"Unknown metric for reduction: {key}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()