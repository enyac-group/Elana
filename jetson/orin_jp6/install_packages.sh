#!/bin/bash
set -euo pipefail

echo "=== JP6 (cu126, Python 3.10) wheel installer ==="

# Directory for downloaded wheels
WHEEL_DIR="wheels_jp6"
mkdir -p "$WHEEL_DIR"
cd "$WHEEL_DIR"

download() {
    local URL="$1"
    echo "[Downloading] $URL"
    wget -q --show-progress "$URL"
}

install_wheel() {
    local WHL_PATTERN="$1"
    local WHL_FILE
    WHL_FILE=$(ls $WHL_PATTERN 2>/dev/null || true)
    if [[ -z "$WHL_FILE" ]]; then
        echo "[ERROR] Wheel not found for pattern: $WHL_PATTERN"
        exit 1
    fi
    echo "[Installing] $WHL_FILE"
    pip install "./$WHL_FILE"
}

echo "=== Step 1: Downloading mandatory packages ==="
# (from https://pypi.jetson-ai-lab.io/jp6/cu126/)

# triton 3.4.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/93c/a9991fe14fa95/triton-3.4.0-cp310-cp310-linux_aarch64.whl"

# pytorch 2.8.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl"

# torchvision 0.23.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"

# torchaudio 2.8.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/de1/5388b8f70e4e1/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl"

# torchcodec 0.6.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/7b5/ac36302315e8d/torchcodec-0.6.0-cp310-cp310-linux_aarch64.whl"

# causal_conv1d 1.5.2
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/42d/974e11e3314ff/causal_conv1d-1.5.2-cp310-cp310-linux_aarch64.whl"

# flash_attn 2.8.4
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/fcd/45bbd956b73b1/flash_attn-2.8.4-cp310-cp310-linux_aarch64.whl"

echo "=== Step 2: Installing mandatory wheels (in dependency order) ==="
# Core torch stack first
install_wheel "torch-2.8.0-*.whl"
install_wheel "torchvision-0.23.0-*.whl"
install_wheel "torchaudio-2.8.0-*.whl"
install_wheel "torchcodec-0.6.0-*.whl"

# Extensions / kernels
install_wheel "triton-3.4.0-*.whl"
install_wheel "causal_conv1d-1.5.2-*.whl"
install_wheel "flash_attn-2.8.4-*.whl"

echo "=== Mandatory package installation complete! ==="

### Optional packages (use: ./install_jp6_wheels.sh --with-optional)
if [[ "${1:-}" == "--with-optional" ]]; then
    echo "=== Step 3: Downloading OPTIONAL packages ==="

    # mamba_ssm 2.2.5
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/c6c/06c8680ffea39/mamba_ssm-2.2.5-cp310-cp310-linux_aarch64.whl"

    # torchao 0.14.0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/7d3/707930a000fd6/torchao-0.14.0-cp39-abi3-linux_aarch64.whl"

    # bitsandbytes 0.48.0.dev0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d46/6b5819e312dd5/bitsandbytes-0.48.0.dev0-cp310-cp310-linux_aarch64.whl"

    # torch_tensorrt 2.8.0+cu126
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/854/fe6d63a2a7526/torch_tensorrt-2.8.0+cu126-cp310-cp310-linux_aarch64.whl"

    # xformers 0.0.33+ac00641.d20250830
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/731/15133b0ebb2b3/xformers-0.0.33+ac00641.d20250830-cp39-abi3-linux_aarch64.whl"

    # vllm 0.10.2+cu126
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/2b9/b377031628a52/vllm-0.10.2+cu126-cp310-cp310-linux_aarch64.whl"

    # flashinfer_python 0.3.1
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/c14/9ad58b9dd019e/flashinfer_python-0.3.1-cp310-cp310-linux_aarch64.whl"

    # nvidia_cutlass 4.0.0.0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/5b9/38c85589f3e81/nvidia_cutlass-4.0.0.0-py3-none-any.whl"

    # onnxruntime_gpu 1.23.0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/e1e/9e3dc2f4d5551/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"

    echo "=== Installing OPTIONAL wheels ==="
    install_wheel "mamba_ssm-2.2.5-*.whl"
    install_wheel "torchao-0.14.0-*.whl"
    install_wheel "bitsandbytes-0.48.0.dev0-*.whl"
    install_wheel "torch_tensorrt-2.8.0+cu126-*.whl"
    install_wheel "xformers-0.0.33+ac00641.d20250830-*.whl"
    install_wheel "vllm-0.10.2+cu126-*.whl"
    install_wheel "flashinfer_python-0.3.1-*.whl"
    install_wheel "nvidia_cutlass-4.0.0.0-*.whl"
    install_wheel "onnxruntime_gpu-1.23.0-*.whl"

    echo "=== Optional package installation complete! ==="
else
    echo "=== Optional packages skipped. Use: --with-optional to include them ==="
fi

echo "=== All done (JP6 stack installed)! ==="
