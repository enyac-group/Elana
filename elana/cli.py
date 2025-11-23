#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Elana launcher")
    parser.add_argument("model_repo", type=str, help="Model repo")
    parser.add_argument("--ngpus", type=int, default=1, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tpot", action="store_true")
    parser.add_argument("--ttft", action="store_true")
    parser.add_argument("--ttlt", action="store_true")
    parser.add_argument("--size", action="store_true")
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--cache_graph", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")

    args, unknown = parser.parse_known_args()

    # Build arguments passed directly to elana/main.py
    profiler_args = [
        args.model_repo,
        f"--batch_size={args.batch_size}",
        f"--prompt_len={args.prompt_len}",
        f"--gen_len={args.gen_len}",
        f"--repeats={args.repeats}",
        f"--log_level={args.log_level}",
    ]

    if args.tpot: profiler_args.append("--tpot")
    if args.ttft: profiler_args.append("--ttft")
    if args.ttlt: profiler_args.append("--ttlt")
    if args.size: profiler_args.append("--size")
    if args.energy: profiler_args.append("--energy")
    if args.cache_graph: profiler_args.append("--cache_graph")

    # Unknown args are forwarded (nice CLI flexibility)
    profiler_args.extend(unknown)

    # Build torchrun command
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.ngpus}",
        os.path.join(os.path.dirname(__file__), "main.py"),
    ] + profiler_args

    print("Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()