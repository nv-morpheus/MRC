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
ARG FROM_IMAGE=nvcr.io/dev/null

# Extend the CI image for development
FROM ${FROM_IMAGE}

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
