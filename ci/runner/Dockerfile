# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_VER=11.4.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8

# Configure the base docker img
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

ARG PROJ_NAME=srf
ARG CUDA_SHORT_VER=11.4
ARG CLANG_VER=12

SHELL ["/bin/bash",  "-c"]

# OS deps
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
        libnuma1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY ./ci/conda/environments/* /tmp/conda/

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n ${PROJ_NAME} -q --file /tmp/conda/dev_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /tmp/conda/clang_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /tmp/conda/ci_env.yml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda


# Install IWYU (remove once we are on clang-14)
RUN git clone https://github.com/include-what-you-use/include-what-you-use.git /tmp/iwyu &&\
    cd /tmp/iwyu &&\
    git checkout clang_${CLANG_VER} &&\
    source activate ${PROJ_NAME} && \
    cmake -G Ninja \
        -DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  . && \
    cmake --build . --parallel $(nproc) --target install && \
    cd - && \
    rm -rf /tmp/iwyu

# ============ driver ==================
FROM base as driver

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y libnvidia-compute-495 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
