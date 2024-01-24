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

MRC_BUILD_TYPE=${MRC_BUILD_TYPE:-Release}

# For now CUDAHOSTCXX is set to `/usr/bin/g++` by
# https://github.com/rapidsai/docker/blob/161b200157206660d88fb02cf69fe58d363ac95e/generated-dockerfiles/rapidsai-core_ubuntu18.04-devel.Dockerfile
# To use GCC-9 in conda build environment, need to set it to $CXX (=$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++)
# This can be removed once we switch to use gcc-9
# : https://docs.rapids.ai/notices/rdn0002/
export CUDAHOSTCXX=${CXX}

export CCACHE_BASEDIR=$(realpath ${SRC_DIR}/..)
# export CCACHE_LOGFILE=${MRC_CACHE_DIR}/ccache/ccache.log
export CCACHE_DEBUG=1
export CCACHE_DEBUGDIR=${SRC_DIR}/ccache_debug
export CCACHE_SLOPPINESS="system_headers"
export CCACHE_NOHASHDIR=1

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)

# This will store all of the cmake args. Make sure to prepend args to allow
# incoming values to overwrite them
CMAKE_ARGS=${CMAKE_ARGS:-""}

# Check for some mrc environment variables. Append to front of args to allow users to overwrite them
if [[ -n "${MRC_CACHE_DIR}" ]]; then
   # Set the cache variable, then set the Staging prefix to allow for host searching
   CMAKE_ARGS="-DMRC_CACHE_DIR=${MRC_CACHE_DIR} ${CMAKE_ARGS}"
fi

# Use the GNU paths to help ccache
export CC=${GCC}
export CXX=${GXX}

# Common CMake args
CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${MRC_BUILD_TYPE} ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES=-"RAPIDS"} ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_INSTALL_LIBDIR=lib ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$PREFIX ${CMAKE_ARGS}"
CMAKE_ARGS="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMRC_BUILD_PYTHON=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DMRC_RAPIDS_VERSION=${rapids_version} ${CMAKE_ARGS}"
CMAKE_ARGS="-DMRC_USE_CCACHE=OFF ${CMAKE_ARGS}"
CMAKE_ARGS="-DMRC_USE_CONDA=ON ${CMAKE_ARGS}"
CMAKE_ARGS="-DPython_EXECUTABLE=${PYTHON} ${CMAKE_ARGS}"

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"
echo "CUDAHOSTCXX : ${CUDAHOSTCXX}"
echo "CUDA        : ${CUDA}"
echo "CMAKE_ARGS  : ${CMAKE_ARGS}"

echo "========Begin Env========"
env
echo "========End Env========"

BUILD_DIR="build-conda"

# Check if the build directory already exists. And if so, delete the
# CMakeCache.txt and CMakeFiles to ensure a clean configuration
if [[ -d "./${BUILD_DIR}" ]]; then
   echo "Deleting old CMake files at ./${BUILD_DIR}"
   rm -rf "./${BUILD_DIR}/CMakeCache.txt"
   rm -rf "./${BUILD_DIR}/CMakeFiles"
fi

echo "PYTHON: ${PYTHON}"
echo "which python: $(which python)"

git submodule update --init --recursive

# Run configure
cmake -B ${BUILD_DIR} \
   ${CMAKE_ARGS} \
   --log-level=verbose \
   .

# Build the components
cmake --build ${BUILD_DIR} -j${PARALLEL_LEVEL:-$(nproc)}
