# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note this is the Dockerfile for the CI runner not for SRF itself

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_VER=11.4.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY ./conda/environments/dev_env.yml /tmp
COPY ./conda/environments/clang_env.yml /tmp

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n srf -q --file /tmp/dev_env.yml && \
    mamba env update -q -n srf --file /tmp/clang_env.yml && \
    # Clean and activate
    # conda clean -afy && \
    sed -i "s/conda activate base/conda activate srf/g" ~/.bashrc && \
    echo -e '#!/bin/sh\n\nexport CUDA_PATH=/usr/local/cuda\nexport LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}' > /opt/conda/envs/srf/etc/conda/activate.d/env_vars.sh && \
    conda clean -afy && \
    rm /tmp/dev_env.yml /tmp/clang_env.yml


# Install IWYU
RUN source activate srf &&\
    git clone https://github.com/include-what-you-use/include-what-you-use.git /tmp/iwyu &&\
    cd /tmp/iwyu &&\
    git checkout clang_12 &&\
    cmake -G Ninja \
        -DCMAKE_PREFIX_PATH=$(llvm-config --cmakedir) \
        -DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  . && \
    cmake --build . --parallel $(nproc) --target install && \
    cd - && \
    rm -rf /tmp/iwyu
