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


ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_VER=11.4.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8

# ============= base ===================
FROM ${FROM_IMAGE}:cuda11.4.1-ubuntu20.04-py3.8 AS base

ARG PROJ_NAME=mrc

SHELL ["/bin/bash",  "-c"]

RUN --mount=type=cache,target=/var/cache/apt \
    apt update &&\
    apt install --no-install-recommends -y \
    libnuma1 && \
    rm -rf /var/lib/apt/lists/*

COPY ./ci/conda/environments/* /opt/mrc/conda/environments/

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    echo "create env: ${PROJ_NAME}" && \
    CONDA_ALWAYS_YES=true \
    /opt/conda/bin/mamba env create -q -n ${PROJ_NAME} --file /opt/mrc/conda/environments/dev_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /opt/mrc/conda/environments/clang_env.yml && \
    /opt/conda/bin/mamba env update -q -n ${PROJ_NAME} --file /opt/mrc/conda/environments/ci_env.yml && \
    chmod -R a+rwX /opt/conda && \
    rm -rf /tmp/conda

RUN /opt/conda/bin/conda init --system &&\
    sed -i 's/xterm-color)/xterm-color|*-256color)/g' ~/.bashrc &&\
    echo "conda activate ${PROJ_NAME}" >> ~/.bashrc

# disable sscache wrappers around compilers
ENV CMAKE_CUDA_COMPILER_LAUNCHER=
ENV CMAKE_CXX_COMPILER_LAUNCHER=
ENV CMAKE_C_COMPILER_LAUNCHER=

# ============ driver ==================
FROM base as driver

RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
    libnvidia-compute-495 \
    && \
    rm -rf /var/lib/apt/lists/*

# ========= development ================
FROM base as development

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update &&\
    apt-get install --no-install-recommends -y \
    gdb \
    htop \
    less \
    openssh-client \
    psmisc \
    sudo \
    vim-tiny \
    && \
    rm -rf /var/lib/apt/lists/*

# Install the .NET SDK. This is a workaround for https://github.com/dotnet/vscode-dotnet-runtime/issues/159
# Once version 1.6.1 of the extension has been release, this can be removed
RUN --mount=type=cache,target=/var/cache/apt \
    wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb &&\
    sudo dpkg -i packages-microsoft-prod.deb &&\
    rm packages-microsoft-prod.deb &&\
    apt-get update && \
    apt-get install --no-install-recommends -y dotnet-sdk-6.0 &&\
    rm -rf /var/lib/apt/lists/*

# create a user inside the container
ARG USERNAME=morpheus
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    usermod --shell /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    cp /root/.bashrc /home/$USERNAME/.bashrc

USER $USERNAME

# default working directory
WORKDIR /work

# Setup git to allow other users to access /work. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development. Utilize --system to not interfere with VS Code
RUN git config --system --add safe.directory "*" && \
    git config --system core.editor "vim"
