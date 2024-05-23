#!/bin/bash
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

export WORKSPACE_TMP="$(pwd)/.tmp/local_ci_workspace"
mkdir -p ${WORKSPACE_TMP}
git clone ${GIT_URL} mrc
cd mrc/
git checkout ${GIT_BRANCH}
git pull
git checkout ${GIT_COMMIT}
git fetch --tags

export MRC_ROOT=$(pwd)
export WORKSPACE=${MRC_ROOT}
export LOCAL_CI=1
GH_SCRIPT_DIR="${MRC_ROOT}/ci/scripts/github"

unset CMAKE_CUDA_COMPILER_LAUNCHER
unset CMAKE_CXX_COMPILER_LAUNCHER
unset CMAKE_C_COMPILER_LAUNCHER

if [[ "${STAGE}" != "bash" ]]; then
    # benchmark & codecov are composite stages, the rest are composed of a single shell script
    if [[ "${STAGE}" == "benchmark" || "${STAGE}" == "codecov" ]]; then
        CI_SCRIPT="${WORKSPACE_TMP}/ci_script.sh"
        echo "#!/bin/bash" > ${CI_SCRIPT}
        if [[ "${STAGE}" == "benchmark" ]]; then
            echo "${GH_SCRIPT_DIR}/pre_benchmark.sh" >> ${CI_SCRIPT}
            echo "${GH_SCRIPT_DIR}/benchmark.sh" >> ${CI_SCRIPT}
            echo "${GH_SCRIPT_DIR}/post_benchmark.sh" >> ${CI_SCRIPT}
        else
            echo "${GH_SCRIPT_DIR}/build.sh" >> ${CI_SCRIPT}
            echo "${GH_SCRIPT_DIR}/test_codecov.sh" >> ${CI_SCRIPT}
        fi

        chmod +x ${CI_SCRIPT}
    else
        if [[ "${STAGE}" =~ "build" ]]; then
            CI_SCRIPT="${GH_SCRIPT_DIR}/build.sh"
        elif [[ "${STAGE}" =~ "test" ]]; then
            CI_SCRIPT="${GH_SCRIPT_DIR}/test.sh"
        else
            CI_SCRIPT="${GH_SCRIPT_DIR}/${STAGE}.sh"
        fi
    fi

    ${CI_SCRIPT}
fi
