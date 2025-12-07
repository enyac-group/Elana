#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-1.5B"
  # "state-spaces/mamba-2.8b-hf"
)

# We could only use ngpus 1 on Jetson Thor, and we don't set CUDA_VISIBLE_DEVICES
WORKLOADS=(
  # ngpus=1, bsz=1, L=512
  "--ttft --tpot --ngpus 1 --batch_size 1 --prompt_len 512 --gen_len 512 --energy --cache_graph --repeats 100"
  "--ttlt           --ngpus 1 --batch_size 1 --prompt_len 512 --gen_len 512 --energy --cache_graph --repeats 20"

  # ngpus=1, bsz=16, L=512
  "--ttft --tpot --ngpus 1 --batch_size 16 --prompt_len 512 --gen_len 512 --energy --cache_graph --repeats 100"
  "--ttlt           --ngpus 1 --batch_size 16 --prompt_len 512 --gen_len 512 --energy --cache_graph --repeats 20"

  # ngpus=1, bsz=16, L=1024
  "--ttft --tpot --ngpus 1 --batch_size 16 --prompt_len 1024 --gen_len 1024 --energy --cache_graph --repeats 100"
  "--ttlt           --ngpus 1 --batch_size 16 --prompt_len 1024 --gen_len 1024 --energy --cache_graph --repeats 20"
)

LOG_DIR="elana_logs/thor"
mkdir -p "${LOG_DIR}"

for model in "${MODELS[@]}"; do
  for workload in "${WORKLOADS[@]}"; do
    # Build a nice log name from model + key args
    model_tag=$(echo "${model}" | tr '/:' '__')
    tag=$(echo "${workload}" | tr ' ' '_' | tr -d '-')
    log_file="${LOG_DIR}/${model_tag}_${tag}.log"

    echo "=================================================================="
    echo "Running: elana ${model} ${workload}"
    echo "Log:     ${log_file}"
    echo "=================================================================="

    # Run and tee to log
    elana "${model}" ${workload} 2>&1 | tee "${log_file}"
  done
done