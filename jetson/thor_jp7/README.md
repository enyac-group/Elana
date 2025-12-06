# Install ELANA on Jetson Thor (Jetpack 7.0)


## 🚀 Create Environment

## Clone repository
```bash
git clone https://github.com/hychiang-git/Elana.git
cd Elana/jetson/thor_jp7
```

### Using virtualenv
We follow the official [document](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#install-multiple-versions-pytorch) and recommand using venv on jetson series.
```bash
python3 -m venv elana-env
source elana-env/bin/activate
pip install --upgrade pip
```

## Install CUDA Toolkit 13.1
Follow the official [document](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html#ways-to-natively-install-cuda-toolkit) to install CUDA Toolkit first.

Remember to add the following lines to your ~/.bashrc file, see [here](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html#post-install-setup-cuda-path-configuration)
```bash
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

- Sanity check
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
 For more details, please read the official [instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform), and the pre-built wheels [arxiv](https://pypi.jetson-ai-lab.io/sbsa/cu130) for Jetson Thor.

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
- libcudnn.so.9
    ```
    ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
    ```
    Install `libcudnn9-cuda-13`
    ```
    sudo apt update
    sudo apt install libcudnn9-cuda-13 libcudnn9-dev-cuda-13
    ```
- libnvpl_lapack_lp64_gomp.so.0
    ```
    ImportError: libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file: No such file or directory
    ```
    Check if `libnvpl_lapack_lp64_gomp.so.0` exists under `/home/{username}/Documents/Elana/elana-env/lib/python3.12/site-packages/nvpl/lib`. If not, try installing `nvpl-lapack` from pip
    ```
    pip install nvpl-lapack
    ```
    Now, you may see `libnvpl_lapack_lp64_gomp.so.0` under the above path. Then, append the path to `LD_LIBRARY_PATH`:
    ```
    export LD_LIBRARY_PATH="/home/{username}/Documents/Elana/elana-env/lib/python3.12/site-packages/nvpl/lib:${LD_LIBRARY_PATH:-}"
    ```
-  libcudss.so.0
    ```
    ImportError: libcudss.so.0: cannot open shared object file: No such file or directory
    ```
    Follow the [instruction](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_local) to download `cudss`: Linux -> arm64-sbsa -> Native -> Ubuntu -> 24.04 -> deb (local). Then, install `cudss`.
- jtop.core.exceptions.JtopException: I can't access jtop.service.
    ```
    PermissionError: [Errno 13] Permission denied

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
        ...
        raise JtopException("I can't access jtop.service.\nPlease logout or reboot this board.")
    jtop.core.exceptions.JtopException: I can't access jtop.service.
    Please logout or reboot this board.
    ```
    1. try to add yourself to the jtop group. Then, run `jtop`
    ```bash
    sudo usermod -a -G jtop $USER
    newgrp jtop
    ``` 
    2. `Mismatch version jtop service`, and check the `jtop` vesion in `elana-env`
    ```bash
    Mismatch version jtop service: [4.5.4] and client: [4.3.2]. Please run:
    sudo systemctl restart jtop.service
    ```
    
    ```bash
    >>> import jtop
    >>> jtop.__version__
    '4.3.2'
    >>> 
    ```

    3. uninstall `jetson-stats` from `elana-env`, and clean the hash table
    ```bash
    pip uninstall jetson-stats
    type jtop      # jtop is hashed (/home/hc29225/Documents/Elana/elana-env/bin/jtop)
    hash -r        # clears the command hash table
    ```

    4. check the system jtop
    ```bash
    $ ls /opt/jtop/venv/bin/python
    /opt/jtop/venv/bin/python
    $ sudo /opt/jtop/venv/bin/python -c "import jtop, site; \
    print('service jtop version:', jtop.__version__); \
    print('site-packages:', [p for p in site.getsitepackages() if 'site-packages' in p][0])"
    service jtop version: 4.5.4
    site-packages: /opt/jtop/venv/lib/python3.12/site-packages
    ```

    5. export the system jtop path
    ```bash
    export PYTHONPATH=/opt/jtop/venv/lib/python3.12/site-packages:$PYTHONPATH
    ```

    6. sanity check the `jtop` in the `elana-env`
    ```bash
    >>> import jtop
    >>> jtop.__version__
    '4.5.4'
    >>> 
    ```

## Install Elana
Finall, we install elana on jetson thor. All dependencies should already be installed
```bash
# cd to Elana root directory, and then install elana
cd ../.. 
pip install .
```