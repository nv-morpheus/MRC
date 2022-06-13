#!/bin/bash
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
mamba env create -n srf -q --file ${SRF_ROOT}/ci/conda/environments/dev_env.yml
conda deactivate
conda activate srf

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version
cmake --version
ninja --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Configuring for build and test"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DSRF_USE_IWYU=ON .

gpuci_logger "Building SRF"
cmake --build build --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for SRF build:"
sccache --show-stats

gpuci_logger "Installing SRF"
pip install ${SRF_ROOT}/build/python

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-editable-packages --ignore-missing-files --n-threads ${PARALLEL_LEVEL} -n morpheus -o ${WORKSPACE_TMP}/conda_env.tar.gz
tar cfj ${WORKSPACE_TMP}/workspace.tar.bz --exclude=".git" --exclude="models" --exclude=".cache" ./
ls -lh ${WORKSPACE_TMP}/

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_env.tar.gz" "${ARTIFACT_URL}/conda_env.tar.gz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/workspace.tar.bz" "${ARTIFACT_URL}/workspace.tar.bz"

gpuci_logger "Success"
exit 0
