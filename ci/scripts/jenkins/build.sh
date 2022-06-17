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

rm -rf ${SRF_ROOT}/.cache/ ${SRF_ROOT}/build/

gpuci_logger "Creating conda env"
mamba env create -n srf -q --file ${CONDA_ENV_YML}
conda deactivate
conda activate srf

mamba env update -q -n srf --file ${SRF_ROOT}/ci/conda/environments/ci_env.yml

CMAKE_CACHE_FLAGS="-DCCACHE_PROGRAM_PATH=$(which sccache) -DSRF_USE_CCACHE=ON"

gpuci_logger "Check versions"
python3 --version
cmake --version
ninja --version

if [[ "${BUILD_CC}" == "gcc" ]]; then
    gpuci_logger "Building with GCC"
    gcc --version
    g++ --version
    CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_CACHE_FLAGS}"
else
    gpuci_logger "Installing Clang"
    mamba install -q -y -c conda-forge "clang=12" "clangxx=12" "libclang=12"
    gpuci_logger "Building with Clang"
    clang --version
    clang++ --version
    CMAKE_CLANG_OPTIONS="-DCMAKE_C_COMPILER:FILEPATH=$(which clang) -DCMAKE_CXX_COMPILER:FILEPATH=$(which clang++) -DCMAKE_CUDA_COMPILER:FILEPATH=$(which nvcc)"
    CMAKE_FLAGS="${CMAKE_CLANG_OPTIONS} ${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_CACHE_FLAGS}"
fi

show_conda_info

gpuci_logger "Configuring for build and test"
cmake -B build -G Ninja ${CMAKE_FLAGS} .

gpuci_logger "Building SRF"
cmake --build build --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for SRF build:"
sccache --show-stats

gpuci_logger "Installing SRF"
cmake -P ${SRF_ROOT}/build/cmake_install.cmake
pip install ${SRF_ROOT}/build/python

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-missing-files --n-threads ${PARALLEL_LEVEL} -n srf -o ${WORKSPACE_TMP}/conda_env.tar.gz
tar cfj "${WORKSPACE_TMP}/cpp_tests.tar.bz" $(find build/ -name "*.x")
tar cfj "${WORKSPACE_TMP}/dsos.tar.bz" $(find build/ -name "*.so")
tar cfj "${WORKSPACE_TMP}/python_build.tar.bz" build/python
ls -lh ${WORKSPACE_TMP}/

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_env.tar.gz" "${ARTIFACT_URL}/conda_env.tar.gz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/cpp_tests.tar.bz" "${ARTIFACT_URL}/cpp_tests.tar.bz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/dsos.tar.bz" "${ARTIFACT_URL}/dsos.tar.bz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/python_build.tar.bz" "${ARTIFACT_URL}/python_build.tar.bz"

gpuci_logger "Success"
