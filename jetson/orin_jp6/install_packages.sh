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
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl#sha256=9da4bcb8e8f0eba00a097ad8c57b26102add499e520d67fb2d5362bebf976ca3"

# pytorch 2.8.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"

# torchvision 0.23.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1"

# torchaudio 2.8.0
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=81a775c8af36ac859fb3f4a1b2f662d5fcf284a835b6bb4ed8d0827a6aa9c0b7"

# causal_conv1d 1.5.2
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/28a/11e19b7f9fd56/causal_conv1d-1.5.2-cp310-cp310-linux_aarch64.whl#sha256=28a11e19b7f9fd56f17347da18fa31e09ad2ac5e61b8ed5653f069cbe7e5177b"

# mamba_ssm 2.2.5
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/b8e/35eeb4d7f0ada/mamba_ssm-2.2.5-cp310-cp310-linux_aarch64.whl#sha256=b8e35eeb4d7f0ada87235c15db0408cded09863bf6798ac451d0f65a6035b4ba"

# flash_attn 2.8.2
download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/c90/358c76ebceadc/flash_attn-2.8.2-cp310-cp310-linux_aarch64.whl#sha256=c90358c76ebceadcd8aef5cf3746ef0026ea05a34688c401f6ab2ee1a6fee19a"

echo "=== Step 2: Installing mandatory wheels (in dependency order) ==="
# Core torch stack first
install_wheel "torch-2.8.0-*.whl"
install_wheel "torchvision-0.23.0-*.whl"
install_wheel "torchaudio-2.8.0-*.whl"

# Extensions / kernels
install_wheel "triton-3.4.0-*.whl"
install_wheel "causal_conv1d-1.5.2-*.whl"
install_wheel "mamba_ssm-2.2.5-*.whl"
install_wheel "flash_attn-2.8.2-*.whl"

echo "=== Mandatory package installation complete! ==="

### Optional packages (use: ./install_jp6_wheels.sh --with-optional)
if [[ "${1:-}" == "--with-optional" ]]; then
    echo "=== Step 3: Downloading OPTIONAL packages ==="

    # torchao 0.13.0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/543/f621c5929ad1a/torchao-0.13.0-cp39-abi3-linux_aarch64.whl#sha256=543f621c5929ad1a942fd9b6728a486e239b5db351b45842c57ce65924ae26b0"

    # bitsandbytes 0.48.0.dev0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/014/eff8ba676c7a3/bitsandbytes-0.47.0.dev0-cp310-cp310-linux_aarch64.whl#sha256=014eff8ba676c7a3830b9430744115af50790d2f7ff1b57f155a8839bcc39104"

    # torch_tensorrt 2.8.0+cu126
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/854/fe6d63a2a7526/torch_tensorrt-2.8.0+cu126-cp310-cp310-linux_aarch64.whl#sha256=854fe6d63a2a75266cf89df5ba6f1dcbe3a6716ed52db86c541fe7483f4199c1"

    # xformers 0.0.33+ac00641.d20250830
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/0a3/cd506d8b3ca50/xformers-0.0.32+8ed0992.d20250724-cp39-abi3-linux_aarch64.whl#sha256=0a3cd506d8b3ca500c79bcb4cbf8cc82e7df844ace08fc5c53192375f7d13085"

    # vllm 0.10.2+cu126
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/9ce/72136a1106950/vllm-0.10.2+cu126-cp310-cp310-linux_aarch64.whl#sha256=9ce72136a1106950aafa5b4fc58dcb73722c295e0ef1e95026e8d23b378b0b6a"

    # flashinfer_python 0.3.1
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/77b/8da187ac5d1aa/flashinfer_python-0.2.9-cp312-cp312-linux_aarch64.whl#sha256=77b8da187ac5d1aa393f80ebf4ccfb7f03f16100a124b5ba614f4dcbaed237c6"

    # onnxruntime_gpu 1.23.0
    download "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"

    echo "=== Installing OPTIONAL wheels ==="
    install_wheel "torchao-0.14.0-*.whl"
    install_wheel "bitsandbytes-0.48.0.dev0-*.whl"
    install_wheel "torch_tensorrt-2.8.0+cu126-*.whl"
    install_wheel "xformers-0.0.33+ac00641.d20250830-*.whl"
    install_wheel "vllm-0.10.2+cu126-*.whl"
    install_wheel "flashinfer_python-0.3.1-*.whl"
    install_wheel "onnxruntime_gpu-1.23.0-*.whl"

    echo "=== Optional package installation complete! ==="
else
    echo "=== Optional packages skipped. Use: --with-optional to include them ==="
fi

echo "=== All done (JP6 stack installed)! ==="
