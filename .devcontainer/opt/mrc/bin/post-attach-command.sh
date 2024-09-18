#!/bin/bash
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

# Ensure our ~/.config directory has the correct permissions. If ~/.config did
# not exist, and you mount ~/.config/gh from the host, then ~/.config will be
# created with root permissions which can break things

conda_env_find(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

ENV_NAME=${ENV_NAME:-mrc}

sed -ri "s/conda activate base/conda activate $ENV_NAME/g" ~/.bashrc;

if conda_env_find "${ENV_NAME}" ; \

then mamba env update --name ${ENV_NAME} -f ${MRC_ROOT}/conda/environments/all_cuda-125_arch-x86_64.yaml --prune; \
else mamba env create --name ${ENV_NAME} -f ${MRC_ROOT}/conda/environments/all_cuda-125_arch-x86_64.yaml; \
fi
