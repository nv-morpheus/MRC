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

gpuci_logger "Check versions"
python3 --version
cmake --version
ninja --version
doxygen --version

show_conda_info

gpuci_logger "Configuring for docs"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DSRF_BUILD_DOCS=ON .


gpuci_logger "Building docs"
cmake --build build --target srf_docs

gpuci_logger "Tarring the docs"
tar cfj "${WORKSPACE_TMP}/docs.tar.bz" build/docs/html

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp --no-progress "${WORKSPACE_TMP}/docs.tar.bz" "${ARTIFACT_URL}/docs.tar.bz"

gpuci_logger "Success"
