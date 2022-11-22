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

rapids-logger "Check versions"
python3 --version
cmake --version
ninja --version
doxygen --version

show_conda_info

rapids-logger "Configuring for docs"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DSRF_BUILD_DOCS=ON .


rapids-logger "Building docs"
cmake --build build --target srf_docs

rapids-logger "Tarring the docs"
tar cfj "${WORKSPACE_TMP}/docs.tar.bz" build/docs/html

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp --no-progress "${WORKSPACE_TMP}/docs.tar.bz" "${ARTIFACT_URL}/docs.tar.bz"

rapids-logger "Success"
