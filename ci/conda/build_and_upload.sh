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

# Color variables
export b="\033[0;36m"
export g="\033[0;32m"
export r="\033[0;31m"
export e="\033[0;90m"
export y="\033[0;33m"
export x="\033[0m"

# Get the path to the root git folder
export SRF_ROOT=${SRF_ROOT:-$(git rev-parse --show-toplevel)}

# Where the conda packages are saved to outside of the conda environment
export CONDA_BLD_OUTPUT=${CONDA_BLD_OUTPUT:-"${SRF_ROOT}/.conda-bld"}

# Set the conda token
CONDA_TOKEN=${CONDA_TOKEN:?"CONDA_TOKEN must be set to allow upload"}

# Get the label to apply to the package
CONDA_PKG_LABEL=${CONDA_PKG_LABEL:-"dev"}

# Ensure we have anaconda-client installed
if [[ -z "$(conda list | grep anaconda-client)" ]]; then
   echo -e "${y}anaconda-client not found. Installing...${x}"

   # Try to keep the dependencies the same
   mamba install -y --no-update-deps anaconda-client
fi

CONDA_ARGS=()

CONDA_ARGS+=("--output-folder=${CONDA_BLD_OUTPUT}")
CONDA_ARGS+=("--token" "${CONDA_TOKEN}")

if [[ -n "${CONDA_PKG_LABEL}" ]]; then
   CONDA_ARGS+=("--label" "${CONDA_PKG_LABEL}")
   echo -e "${y}Uploading with label: ${CONDA_PKG_LABEL}${x}"
fi

echo -e "${b}=====Beginning Package Build/Upload=====${x}"

CONDA_ARGS="${CONDA_ARGS[@]}" ${SRF_ROOT}/ci/conda/recipes/run_conda_build.sh

echo -e "${b}=====Beginning Package Build/Upload Complete=====${x}"
