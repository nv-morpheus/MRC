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


restore_conda_env

gpuci_logger "Building Conda Package"
CONDA_BLD_OUTPUT="${WORKSPACE_TMP}/conda-bld"
mkdir -p ${CONDA_BLD_OUTPUT}

CONDA_ARGS=()
CONDA_ARGS+=("--output-folder=${CONDA_BLD_OUTPUT}")
CONDA_ARGS+=("--label" "${CONDA_PKG_LABEL}")
CONDA_ARGS="${CONDA_ARGS[@]}" ${SRF_ROOT}/ci/conda/recipes/run_conda_build.sh

gpuci_logger "Archiving Conda Package"
cd $(dirname ${CONDA_BLD_OUTPUT})
tar cfj ${WORKSPACE_TMP}/conda_pkg.tar.bz $(basename ${CONDA_BLD_OUTPUT})

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp ${WORKSPACE_TMP}/conda_pkg.tar.bz "${ARTIFACT_URL}/conda_pkg.tar.bz"
