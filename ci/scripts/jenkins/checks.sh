#!/usr/bin/bash
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

set -e

source ${WORKSPACE}/ci/scripts/jenkins/common.sh
export IWYU_DIR="${WORKSPACE_TMP}/iwyu"

rm -rf ${SRF_ROOT}/.cache/ ${SRF_ROOT}/build/ ${IWYU_DIR}

fetch_base_branch

gpuci_logger "Creating conda env"
mamba env create -n srf -q --file ${CONDA_ENV_YML}

gpuci_logger "Installing Clang"
mamba env update -q -n srf --file ${SRF_ROOT}/ci/conda/environments/clang_env.yml

conda deactivate
conda activate srf

show_conda_info

gpuci_logger "Installing IWYU"
git clone https://github.com/include-what-you-use/include-what-you-use.git ${IWYU_DIR}
pushd ${IWYU_DIR}
git checkout clang_12
cmake -G Ninja \
    -DCMAKE_PREFIX_PATH=$(llvm-config --cmakedir) \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    .

cmake --build . --parallel ${PARALLEL_LEVEL} --target install

popd

gpuci_logger "Configuring CMake"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} .

gpuci_logger "Building targets that generate source code"
cmake --build build --target srf_style_checks --parallel ${PARALLEL_LEVEL}

gpuci_logger "Running C++ style checks"
${SRF_ROOT}/ci/scripts/cpp_checks.sh

gpuci_logger "Runing Python style checks"
${SRF_ROOT}/ci/scripts/python_checks.sh

gpuci_logger "Checking copyright headers"
python ${SRF_ROOT}/ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits ${CHANGE_TARGET} ${GIT_COMMIT}
