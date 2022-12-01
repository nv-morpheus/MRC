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

ARG PROJ_NAME=mrc
ARG CUDA_SHORT_VER=11.4

SHELL ["/bin/bash",  "-c"]

# OS deps
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
        libnuma1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY ./ci/conda/environments/* /tmp/conda/

RUN CONDA_ALWAYS_YES=true \
    /opt/conda/bin/mamba env create -q -n ${PROJ_NAME} --file /tmp/conda/dev_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /tmp/conda/clang_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /tmp/conda/ci_env.yml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda

# ============ driver ==================
FROM base as driver

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y libnvidia-compute-495 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# ========= development ================
FROM base as dev

# OS deps
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
        git \
        gdb \
        htop \
        openssh-client \
        psmisc \
        vim-tiny \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# default working directory
WORKDIR /work

# Setup git to allow other users to access /work. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"
