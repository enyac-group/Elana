#!/bin/bash
set -euo pipefail

# All pre-built wheels can be found at: https://pypi.jetson-ai-lab.io/sbsa/cu130
echo "=== Jetson Thor wheel installer (CUDA 13.0, Python 3.12) ==="

# Directory for downloaded wheels
WHEEL_DIR="wheels_jp7"
mkdir -p $WHEEL_DIR
cd $WHEEL_DIR

download() {
    URL="$1"
    echo "[Downloading] $URL"
    wget -q --show-progress "$URL"
}

install_wheel() {
    WHL_PATTERN="$1"
    WHL_FILE=$(ls $WHL_PATTERN 2>/dev/null || true)
    if [[ -z "$WHL_FILE" ]]; then
        echo "[ERROR] Wheel not found: $WHL_PATTERN"
        exit 1
    fi
    echo "[Installing] $WHL_FILE"
    pip install "./$WHL_FILE"
}

echo "=== Step 1: Downloading mandatory packages ==="

# Triton 3.5.0
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/968/29c4f038c9bbd/triton-3.5.0-cp312-cp312-linux_aarch64.whl"

# PyTorch 2.9.0
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/d03/870b7c360cc90/torch-2.9.0-cp312-cp312-linux_aarch64.whl"

# Torchaudio 2.9.0
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/57a/b13c699f333fc/torchaudio-2.9.0-cp312-cp312-linux_aarch64.whl"

# TorchCodec 0.8.0
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/dc3/d21e5468f0163/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"

# causal_conv1d 1.5.3
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/9a7/5d9ca74e175e4/causal_conv1d-1.5.3-cp312-cp312-linux_aarch64.whl"

# flash-attn 2.8.4
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/d98/079d4461bfefc/flash_attn-2.8.4-cp312-cp312-linux_aarch64.whl"

# mamba_ssm 2.2.6
download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/077/95469a1a4ccf4/mamba_ssm-2.2.6.post2-cp312-cp312-linux_aarch64.whl"


echo "=== Step 2: Installing mandatory wheels (in dependency order) ==="

install_wheel "torch-2.9.0-*.whl"
install_wheel "torchaudio-2.9.0-*.whl"
install_wheel "torchcodec-0.8.0-*.whl"

install_wheel "triton-3.5.0-*.whl"
install_wheel "causal_conv1d-1.5.3-*.whl"
install_wheel "flash_attn-2.8.4-*.whl"
install_wheel "mamba_ssm-2.2.6.post2-*.whl"


echo "=== Mandatory package installation complete! ==="

### Optional packages (use: ./install_jetson_thor_pkgs.sh --with-optional)
if [[ "${1:-}" == "--with-optional" ]]; then
    echo "=== Step 3: Downloading OPTIONAL packages ==="

    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/468/96491592d19ee/xformers-0.0.33.post1-cp39-abi3-linux_aarch64.whl"
    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/45a/a44d97cd2b49c/vllm-0.11.2+cu130-cp312-cp312-linux_aarch64.whl"
    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/5c5/d04f9cce69a74/flashinfer_python-0.5.2-py3-none-any.whl"
    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/fd1/f48ecc4345d25/nvidia_cutlass-4.2.1.0-py3-none-any.whl"
    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/5fe/5f3eb4e280bd2/torchao-0.13.0-cp39-abi3-linux_aarch64.whl"
    download "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/8d3/a7ce3521d7acf/bitsandbytes-0.48.0-cp312-cp312-linux_aarch64.whl"

    echo "=== Installing OPTIONAL wheels ==="
    install_wheel "xformers-0.0.33.post1-*.whl"
    install_wheel "vllm-0.11.2+cu130-*.whl"
    install_wheel "flashinfer_python-0.5.2-*.whl"
    install_wheel "nvidia_cutlass-4.2.1.0-*.whl"
    install_wheel "bitsandbytes-0.48.0-*.whl"
    install_wheel "torchao-0.13.0-*.whl"

    echo "=== Optional package installation complete! ==="
else
    echo "=== Optional packages skipped. Use: --with-optional to include them ==="
fi

echo "=== All done! ==="
