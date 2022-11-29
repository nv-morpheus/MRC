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

fetch_base_branch

update_conda_env

rapids-logger "Configuring CMake"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} .

rapids-logger "Building targets that generate source code"
cmake --build build --target mrc_style_checks --parallel ${PARALLEL_LEVEL}

rapids-logger "Running C++ style checks"
${MRC_ROOT}/ci/scripts/cpp_checks.sh

rapids-logger "Runing Python style checks"
${MRC_ROOT}/ci/scripts/python_checks.sh

rapids-logger "Checking copyright headers"
python ${MRC_ROOT}/ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits ${CHANGE_TARGET} ${GIT_COMMIT}
