# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


ARG FROM_IMAGE="rapidsai/ci-conda"
ARG CUDA_VER=12.8.0
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=22.04
ARG PYTHON_VER=3.10
ARG REAL_ARCH=notset

# ============= base ===================
FROM --platform=$TARGETPLATFORM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

ARG PROJ_NAME=mrc
ARG USERNAME=morpheus
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG REAL_ARCH

SHELL ["/bin/bash",  "-c"]
ENV REAL_ARCH=${REAL_ARCH}

RUN --mount=type=cache,target=/var/cache/apt,id=apt_cache-${REAL_ARCH} \
    apt update &&\
    apt install --no-install-recommends -y \
    libnuma1 \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# create a user inside the container
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    usermod --shell /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

COPY ./conda/environments/all_cuda-128_arch-${REAL_ARCH}.yaml /opt/mrc/conda/environments/all_cuda-128_arch-${REAL_ARCH}.yaml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked,id=conda_cache-${REAL_ARCH} \
    echo "create env: ${PROJ_NAME}" && \
    sudo -g conda -u $USERNAME \
    CONDA_ALWAYS_YES=true \
    /opt/conda/bin/conda env create --solver=libmamba -q -n ${PROJ_NAME} \
        --file /opt/mrc/conda/environments/all_cuda-128_arch-${REAL_ARCH}.yaml && \
    chmod -R a+rwX /opt/conda && \
    rm -rf /tmp/conda

RUN /opt/conda/bin/conda init --system &&\
    sed -i 's/xterm-color)/xterm-color|*-256color)/g' ~/.bashrc &&\
    echo "conda activate ${PROJ_NAME}" >> ~/.bashrc && \
    cp /root/.bashrc /home/$USERNAME/.bashrc

# disable sscache wrappers around compilers
ENV CMAKE_CUDA_COMPILER_LAUNCHER=
ENV CMAKE_CXX_COMPILER_LAUNCHER=
ENV CMAKE_C_COMPILER_LAUNCHER=

# ============ build ==================
FROM --platform=$TARGETPLATFORM base as build

# Add any build only dependencies here. For now there is none but we need the
# target to get the CI runner build scripts to work

# ============ test ==================
FROM --platform=$TARGETPLATFORM base as test

# Add any test only dependencies here. For now there is none but we need the
# target to get the CI runner build scripts to work

# ========= development ================
FROM --platform=$TARGETPLATFORM base as development
ARG REAL_ARCH

RUN --mount=type=cache,target=/var/cache/apt,id=apt_cache-${REAL_ARCH} \
    apt-get update &&\
    apt-get install --no-install-recommends -y \
    gdb \
    htop \
    less \
    openssh-client \
    psmisc \
    vim-tiny \
    && \
    rm -rf /var/lib/apt/lists/*

# Install the .NET SDK. This is a workaround for https://github.com/dotnet/vscode-dotnet-runtime/issues/159
# Once version 1.6.1 of the extension has been release, this can be removed
RUN --mount=type=cache,target=/var/cache/apt,id=apt_cache-${REAL_ARCH} \
    wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb &&\
    sudo dpkg -i packages-microsoft-prod.deb &&\
    rm packages-microsoft-prod.deb &&\
    apt-get update && \
    apt-get install --no-install-recommends -y dotnet-sdk-6.0 &&\
    rm -rf /var/lib/apt/lists/*

USER $USERNAME

# default working directory
WORKDIR /work

# Setup git to allow other users to access /work. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development. Utilize --system to not interfere with VS Code
RUN git config --system --add safe.directory "*" && \
    git config --system core.editor "vim"
