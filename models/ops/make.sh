#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Use GCC-12 as host compiler if available (CUDA 12.x officially supports up to GCC 12).
# On Ubuntu/Debian: sudo apt install gcc-12 g++-12
if command -v gcc-12 &>/dev/null && command -v g++-12 &>/dev/null; then
  export CC=gcc-12
  export CXX=g++-12
fi

# TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME='/path/to/your/cuda/dir'
python setup.py build install
