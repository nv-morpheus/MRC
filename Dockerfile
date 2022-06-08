# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG FROM_IMAGE="gpuci/miniforge-cuda"
ARG CUDA_VER=11.4
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:${CUDA_VER}-devel-${LINUX_DISTRO}${LINUX_VER} AS base

ARG CONDA_ENV_NAME=srf
ARG PYTHON_VER=3.8

# Update and install some base dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install --no-install-recommends -y \
        build-essential openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    conda config --env --add channels conda-forge &&\
    /opt/conda/bin/conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" python=${PYTHON_VER}
    # conda clean -afy

# All code will be under /work
WORKDIR /work

# ============ Stage: conda_env ============
# Create the conda environment and configure container for running in environment
FROM base as conda_env

ARG CONDA_ENV_NAME

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n ${CONDA_ENV_NAME} -c conda-forge python=${PYTHON_VER} &&\
    # Clean and activate
    # conda clean -afy && \
    sed -i "s/conda activate base/conda activate ${CONDA_ENV_NAME}/g" ~/.bashrc

# Set the permenant conda channes to use for the conda environment
RUN source activate ${CONDA_ENV_NAME} &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY ci/conda/entrypoint.sh ./ci/conda/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "ci/conda/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# ============ Stage: conda_bld_srf ============
# Now build the conda dependency packages
FROM base as conda_bld_srf

# Copy the source
COPY . ./

RUN --mount=type=ssh \
    --mount=type=cache,id=workspace_cache,target=/work/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate base &&\
    SRF_ROOT=/work CONDA_BLD_DIR=/opt/conda/conda-bld CONDA_ARGS="--no-test" ./ci/conda/recipes/run_conda_build.sh

# ============ Stage: runtime ============
# Setup container for runtime environment
FROM conda_env as runtime

RUN --mount=type=bind,from=conda_bld_srf,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate ${CONDA_ENV_NAME} &&\
    # Install conda packages
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n ${CONDA_ENV_NAME} -c local -c rapidsai -c nvidia -c conda-forge srf &&\
    # Clean and activate
    conda clean -afy

# Only copy specific files/folders over that are necessary for runtime
COPY "./docs" "./docs"

# ============ Stage: development ============
# Setup container for development environment
FROM conda_env as development

# Install some base requirements. Install clang-12 and link all of the binaries (ugly)
RUN apt update &&\
    apt-get install -y clang-tools-12 clang-format-12 clang-tidy-12 clangd-12 libclang-12-dev &&\
    rm -rf /var/lib/apt/lists/* &&\
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 121 \
        --slave /usr/bin/clang-apply-replacements clang-apply-replacements /usr/bin/clang-apply-replacements-12 \
        --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-12 \
        --slave /usr/bin/clang-format-diff clang-format-diff /usr/bin/clang-format-diff-12 \
        --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-12 \
        --slave /usr/bin/clang-tidy-diff.py clang-tidy-diff.py /usr/bin/clang-tidy-diff-12.py \
        --slave /usr/bin/clang++ clang++ /usr/bin/clang++-12 \
        --slave /usr/bin/clangd clangd /usr/bin/clangd-12 \
        --slave /usr/bin/git-clang-format git-clang-format /usr/bin/git-clang-format-12 \
        --slave /usr/bin/llvm-symbolizer llvm-symbolizer /usr/bin/llvm-symbolizer-12 \
        --slave /usr/share/man/man1/clang.1.gz clang.1.gz /usr/share/man/man1/clang-12.1.gz

COPY ci/conda/environments/dev_env.yml ./ci/conda/environments/

# Install the dev dependencies
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    /opt/conda/bin/mamba env update -n ${CONDA_ENV_NAME} --file ci/conda/environments/dev_env.yml &&\
    # Clean and activate
    conda clean -afy

# Install IWYU (must be after dependencies for cmake)
RUN source activate ${CONDA_ENV_NAME} &&\
    git clone https://github.com/include-what-you-use/include-what-you-use.git /opt/iwyu &&\
    cd /opt/iwyu &&\
    git checkout clang_12 &&\
    cmake -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib . &&\
    cmake --build . --target install &&\
    echo 'export PATH=$PATH:/opt/iwyu' >>~/.bashrc

# Setup git to allow other users to access /workspace. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"
