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

# Note this is the Dockerfile for the CI runner not for SRF itself
# Build from the root of the SRF repo with:
# docker build --network=host -t srf/ci:latest -f ci/runner/Dockerfile .

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

SHELL ["/bin/bash",  "-c"]

# OS deps
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
        clang-format-12 \
        clang-tidy-12 \
        libclang-12-dev \
        libnuma1 \
        libnvidia-compute-495 \
        llvm-12-dev \
    && rm -rf /var/lib/apt/lists/* &&\
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 121 \
        --slave /usr/bin/clang-apply-replacements clang-apply-replacements /usr/bin/clang-apply-replacements-12 \
        --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-12 \
        --slave /usr/bin/clang-format-diff clang-format-diff /usr/bin/clang-format-diff-12 \
        --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-12 \
        --slave /usr/bin/clang-tidy-diff.py clang-tidy-diff.py /usr/bin/clang-tidy-diff-12.py \
        --slave /usr/bin/clang++ clang++ /usr/bin/clang++-12 \
        --slave /usr/bin/git-clang-format git-clang-format /usr/bin/git-clang-format-12

# Create conda environment
COPY ./ci/conda/environments/dev_env.yml /tmp

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n ${PROJ_NAME} -q --file /tmp/dev_env.yml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm /tmp/dev_env.yml


# Install IWYU
RUN git clone https://github.com/include-what-you-use/include-what-you-use.git /tmp/iwyu &&\
    cd /tmp/iwyu &&\
    git checkout clang_12 &&\
    source activate ${PROJ_NAME} && \
    cmake -G Ninja \
        -DCMAKE_PREFIX_PATH=$(llvm-config --cmakedir) \
        -DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  . && \
    cmake --build . --parallel $(nproc) --target install && \
    cd - && \
    rm -rf /tmp/iwyu
