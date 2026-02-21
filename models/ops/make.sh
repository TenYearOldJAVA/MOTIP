#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Use GCC-11 as host compiler if available (avoids "unsupported GNU version" and glibc 2.38+ _Float* issues).
# On Ubuntu/Debian: sudo apt install gcc-11 g++-11
if command -v gcc-11 &>/dev/null && command -v g++-11 &>/dev/null; then
  export CC=gcc-11
  export CXX=g++-11
fi

# TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME='/path/to/your/cuda/dir'
python setup.py build install

# If you see "_Float32 is undefined" (glibc 2.38+), build inside Ubuntu 22.04 Docker instead.
