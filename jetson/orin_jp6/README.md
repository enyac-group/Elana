# Install ELANA on Jetson Orin (Jetpack 6.2)


## 🚀 Create Environment

## Clone repository
```bash
git clone https://github.com/hychiang-git/Elana.git
cd Elana/jetson/orin_jp6
```

### Using virtualenv
We follow the official [document](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#install-multiple-versions-pytorch) and recommand using venv on jetson series.
```bash
python3 -m venv elana-env
source elana-env/bin/activate
pip install --upgrade pip
```

- Sanity check CUDA Toolkits
```bash
nvcc -V
# you should see
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_07:24:07_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0
```


## Install Dependencies

1. Install packages in the requirements.txt
```bash
# the packages in the requirements.txt can be installed directly on Jetson Thor without compilation
pip install -r requirements.txt
```

2. Install the pre-built wheels for Jetson Thor.
```
bash install_packages.sh
```
 For more details, please read the official [instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform), and the pre-built wheels [arxiv](https://pypi.jetson-ai-lab.io/jp6/cu126/) for Jetson Thor.

3. A sanity check for all dependencies
```
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.__version__
'2.9.0'
>>> import triton
>>> triton.__version__
'3.5.0'
>>> import flash_attn
>>> flash_attn.__version__
'2.8.4'
>>> import mamba_ssm
>>> mamba_ssm.__version__
'2.2.6.post2'
>>> import causal_conv1d
>>> causal_conv1d.__version__
'1.5.3'
>>> 
```

## Troubleshooting
-  libcudss.so.0
    ```
    ImportError: libcudss.so.0: cannot open shared object file: No such file or directory
    ```
    Follow the [instruction](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) to download `cudss`: Linux -> aarch64-jetson -> Native -> Ubuntu -> 22.04 -> deb (local). Then, install `cudss`.



## Install Elana
Finall, we install elana on jetson thor. All dependencies should already be installed
```bash
# cd to Elana root directory, and then install elana
cd ../.. 
pip install .
```