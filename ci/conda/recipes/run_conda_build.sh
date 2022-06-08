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

function get_version() {
   echo "$(git describe --tags | grep -o -E '^([^-]*?)')"
}

# Color variables
export b="\033[0;36m"
export g="\033[0;32m"
export r="\033[0;31m"
export e="\033[0;90m"
export y="\033[0;33m"
export x="\033[0m"

# Change this to switch between build/mambabuild/debug
export CONDA_COMMAND=${CONDA_COMMAND:-"mambabuild"}

# Get the path to the morpheus git folder
export SRF_ROOT=${SRF_ROOT:-$(git rev-parse --show-toplevel)}

export CUDA="$(conda list | grep cudatoolkit | egrep -o "[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+")"
export PYTHON_VER="$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")"
export CUDA=11.4.1
echo "CUDA       : ${CUDA}"
echo "PYTHON_VER : ${PYTHON_VER}"
echo ""

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)}

# Export variables for the cache
export SRF_CACHE_DIR=${SRF_CACHE_DIR:-"${SRF_ROOT}/.cache"}

# Export CCACHE variables
export CCACHE_DIR="${SRF_CACHE_DIR}/ccache"
export CCACHE_NOHASHDIR=1
export CMAKE_GENERATOR="Ninja"
export CMAKE_C_COMPILER_LAUNCHER="ccache"
export CMAKE_CXX_COMPILER_LAUNCHER="ccache"
export CMAKE_CUDA_COMPILER_LAUNCHER="ccache"

# Holds the arguments in an array to allow for complex json objects
CONDA_ARGS_ARRAY=()

# Some default args
CONDA_ARGS_ARRAY+=("--use-local")

if [[ "${CONDA_COMMAND}" == "mambabuild" || "${CONDA_COMMAND}" == "build" ]]; then
   # Remove the timestamp from the work folder to allow caching to work better
   CONDA_ARGS_ARRAY+=("--build-id-pat" "{n}-{v}")
fi

# Choose default variants
CONDA_ARGS_ARRAY+=("--variants" "{python: 3.8}")

# And default channels
CONDA_ARGS_ARRAY+=("-c" "rapidsai" "-c" "nvidia" "-c" "conda-forge" "-c" "main")

# Set GIT_VERSION to set the project version inside of meta.yaml
export GIT_VERSION="$(get_version)"

echo -e "${y}===Begin Env===${x}"
env
echo -e "${y}===End Env===${x}"

echo -e "${y}===Running conda-build for libsrf===${x}"
set -x
conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/libsrf
set +x
echo -e "${g}===Running conda-build for libsrf Complete!===${x}"
