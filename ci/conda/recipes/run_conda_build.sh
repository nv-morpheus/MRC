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

NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

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
export MRC_ROOT=${MRC_ROOT:-$(git rev-parse --show-toplevel)}

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)}

# Export variables for the cache
export MRC_CACHE_DIR=${MRC_CACHE_DIR:-"${MRC_ROOT}/.cache"}

# Export CCACHE variables
export CCACHE_DIR="${MRC_CACHE_DIR}/ccache"
export CCACHE_NOHASHDIR=1
export CMAKE_GENERATOR="Ninja"
export CMAKE_C_COMPILER_LAUNCHER="ccache"
export CMAKE_CXX_COMPILER_LAUNCHER="ccache"
export CMAKE_CUDA_COMPILER_LAUNCHER="ccache"

# Avoid confirmation messages during the conda build
export CONDA_ALWAYS_YES=true

# Holds the arguments in an array to allow for complex json objects
CONDA_ARGS_ARRAY=()

if hasArg upload; then
   # Set the conda token
   CONDA_TOKEN=${CONDA_TOKEN:?"CONDA_TOKEN must be set to allow upload"}

   # Get the label to apply to the package
   CONDA_PKG_LABEL=${CONDA_PKG_LABEL:-"dev"}

   # Ensure we have anaconda-client installed for upload
   if [[ -z "$(conda list | grep anaconda-client)" ]]; then
      echo -e "${y}anaconda-client not found and is required for up. Installing...${x}"

      mamba install -y anaconda-client
   fi

   echo -e "${y}Uploading MRC Conda Package${x}"

   # Add the conda token needed for uploading
   CONDA_ARGS_ARRAY+=("--token" "${CONDA_TOKEN}")

   if [[ -n "${CONDA_PKG_LABEL}" ]]; then
      CONDA_ARGS_ARRAY+=("--label" "${CONDA_PKG_LABEL}")
      echo -e "${y}   Using label: ${CONDA_PKG_LABEL}${x}"
   fi
fi

# Some default args
CONDA_ARGS_ARRAY+=("--use-local")

if [[ "${CONDA_COMMAND}" == "mambabuild" || "${CONDA_COMMAND}" == "build" ]]; then
   # Remove the timestamp from the work folder to allow caching to work better
   CONDA_ARGS_ARRAY+=("--build-id-pat" "{n}-{v}")
fi

# Choose default variants
if hasArg quick; then
   # For quick build, just do most recent version of rapids
   CONDA_ARGS_ARRAY+=("--variants" "{rapids_version: 25.02}")
fi

# And default channels (should match dependencies.yaml)
CONDA_ARGS_ARRAY+=("-c" "conda-forge" "-c" "rapidsai" "-c" "rapidsai-nightly" "-c" "nvidia")

# Set GIT_VERSION to set the project version inside of meta.yaml
export GIT_VERSION="$(get_version)"

echo -e "${y}===Begin Env===${x}"
env
echo -e "${y}===End Env===${x}"

echo -e "${y}===Running conda-build for libmrc===${x}"
set -x
conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/libmrc
set +x
echo -e "${g}===Running conda-build for libmrc Complete!===${x}"
