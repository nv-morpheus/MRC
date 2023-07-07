#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Its important that we are in the base environment for the build
rapids-logger "Activating Base Conda Environment"

# Deactivate any extra environments (There can be a few on the stack)
while [[ "${CONDA_SHLVL:-0}" -gt 1 ]]; do
   echo "Deactivating conda environment ${CONDA_DEFAULT_ENV}"
   conda deactivate
done

# Ensure at least base is activated
if [[ "${CONDA_DEFAULT_ENV}" != "base" ]]; then
   echo "Activating base conda environment"
   conda activate base
fi

# Print the info just to be sure base is active
conda info

rapids-logger "Building Conda Package"

# Run the conda build and upload
${MRC_ROOT}/ci/conda/recipes/run_conda_build.sh "$@"
