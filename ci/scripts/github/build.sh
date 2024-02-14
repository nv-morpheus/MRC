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

source ${WORKSPACE}/ci/scripts/github/common.sh

update_conda_env

if [[ "${LOCAL_CI}" == "" ]]; then
    CMAKE_CACHE_FLAGS="-DCCACHE_PROGRAM_PATH=$(which sccache) -DMRC_USE_CCACHE=ON"
else
    CMAKE_CACHE_FLAGS=""
fi


rapids-logger "Check versions"
python3 --version
cmake --version
ninja --version
sccache --version

if [[ "${BUILD_CC}" == "gcc" ]]; then
    rapids-logger "Building with GCC"
    x86_64-conda-linux-gnu-cc --version
    x86_64-conda-linux-gnu-c++ --version
    CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_CACHE_FLAGS}"
elif [[ "${BUILD_CC}" == "gcc-coverage" ]]; then
    rapids-logger "Building with GCC with gcov profile '-g -fprofile-arcs -ftest-coverage"
    x86_64-conda-linux-gnu-cc --version
    x86_64-conda-linux-gnu-c++ --version
    x86_64-conda-linux-gnu-gcov --version
    CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_BUILD_WITH_CODECOV} ${CMAKE_CACHE_FLAGS}"
else
    rapids-logger "Building with Clang"
    clang --version
    clang++ --version
    CMAKE_CLANG_OPTIONS="-DCMAKE_C_COMPILER:FILEPATH=$(which clang) -DCMAKE_CXX_COMPILER:FILEPATH=$(which clang++) -DCMAKE_CUDA_COMPILER:FILEPATH=$(which nvcc)"
    CMAKE_FLAGS="${CMAKE_CLANG_OPTIONS} ${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_CACHE_FLAGS}"
fi

show_conda_info

rapids-logger "Configuring for build and test"
git submodule update --init --recursive
cmake -B build -G Ninja ${CMAKE_FLAGS} --debug-find-pkg=spdlog --debug-find-pkg=Thrust .

rapids-logger "Building MRC"
cmake --build build --parallel ${PARALLEL_LEVEL}

if [[ "${LOCAL_CI}" == "" ]]; then
    rapids-logger "sccache usage for MRC build:"
    sccache --show-stats
fi

if [[ "${BUILD_CC}" != "gcc-coverage" || ${LOCAL_CI} == "1" ]]; then
    rapids-logger "Archiving results"
    tar cfj "${WORKSPACE_TMP}/dot_cache.tar.bz" .cache
    tar cfj "${WORKSPACE_TMP}/build.tar.bz" build
    ls -lh ${WORKSPACE_TMP}/

    rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
    upload_artifact "${WORKSPACE_TMP}/build.tar.bz"
    upload_artifact "${WORKSPACE_TMP}/dot_cache.tar.bz"
fi

rapids-logger "Success"
