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

rapids-logger "Env Setup"
source /opt/conda/etc/profile.d/conda.sh
export MRC_ROOT=${MRC_ROOT:-$(git rev-parse --show-toplevel)}
cd ${MRC_ROOT}
# For non-gpu hosts nproc will correctly report the number of cores we are able to use
# On a GPU host however nproc will report the total number of cores and PARALLEL_LEVEL
# will be defined specifying the subset we are allowed to use.
NUM_CORES=$(nproc)
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-${NUM_CORES}}
rapids-logger "Procs: ${NUM_CORES}"
rapids-logger "Memory"

/usr/bin/free -g

rapids-logger "user info"
id

# NUM_PROC is used by some of the other scripts
export NUM_PROC=${PARALLEL_LEVEL:-$(nproc)}

export CONDA_ENV_YML="${MRC_ROOT}/ci/conda/environments/dev_env.yml"

export CMAKE_BUILD_ALL_FEATURES="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMRC_BUILD_BENCHMARKS=ON -DMRC_BUILD_EXAMPLES=ON -DMRC_BUILD_PYTHON=ON -DMRC_BUILD_TESTS=ON -DMRC_USE_CONDA=ON -DMRC_PYTHON_BUILD_STUBS=ON"
export CMAKE_BUILD_WITH_CODECOV="-DCMAKE_BUILD_TYPE=Debug -DMRC_ENABLE_CODECOV=ON"

# Set the depth to allow git describe to work
export GIT_DEPTH=1000

# For PRs, $GIT_BRANCH is like: pull-request/989
REPO_NAME=$(basename "${GITHUB_REPOSITORY}")
ORG_NAME="${GITHUB_REPOSITORY_OWNER}"
PR_NUM="${GITHUB_REF_NAME##*/}"


# S3 vars
export S3_URL="s3://rapids-downloads/ci/srf"
export DISPLAY_URL="https://downloads.rapids.ai/ci/srf"
export ARTIFACT_ENDPOINT="/pull-request/${PR_NUM}/${GIT_COMMIT}/${NVARCH}/${BUILD_CC}"
export ARTIFACT_URL="${S3_URL}${ARTIFACT_ENDPOINT}"
export DISPLAY_ARTIFACT_URL="${DISPLAY_URL}${ARTIFACT_ENDPOINT}"

# Set sccache env vars
export SCCACHE_S3_KEY_PREFIX=mrc-${NVARCH}-${BUILD_CC}
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION="${AWS_DEFAULT_REGION}"
export SCCACHE_IDLE_TIMEOUT=32768
#export SCCACHE_LOG=debug

mkdir -p ${WORKSPACE_TMP}

function print_env_vars() {
    rapids-logger "Environ:"
    env | grep -v -E "AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|TOKEN" | sort
}

function update_conda_env() {
    rapids-logger "Checking for updates to conda env"
    rapids-mamba-retry env update -n mrc -q --file ${CONDA_ENV_YML}
    conda deactivate
    conda activate mrc
}

print_env_vars

function fetch_base_branch() {
    rapids-logger "Retrieving base branch from GitHub API"
    [[ -n "$GH_TOKEN" ]] && CURL_HEADERS=('-H' "Authorization: token ${GH_TOKEN}")
    RESP=$(
    curl -s \
        -H "Accept: application/vnd.github.v3+json" \
        "${CURL_HEADERS[@]}" \
        "${GITHUB_API_URL}/repos/${ORG_NAME}/${REPO_NAME}/pulls/${PR_NUM}"
    )

    BASE_BRANCH=$(echo "${RESP}" | jq -r '.base.ref')

    # Change target is the branch name we are merging into but due to the weird way jenkins does
    # the checkout it isn't recognized by git without the origin/ prefix
    export CHANGE_TARGET="origin/${BASE_BRANCH}"
    rapids-logger "Base branch: ${BASE_BRANCH}"
}

function fetch_s3() {
    ENDPOINT=$1
    DESTINATION=$2
    if [[ "${USE_S3_CURL}" == "1" ]]; then
        curl -f "${DISPLAY_URL}${ENDPOINT}" -o "${DESTINATION}"
        FETCH_STATUS=$?
    else
        aws s3 cp --no-progress "${S3_URL}${ENDPOINT}" "${DESTINATION}"
        FETCH_STATUS=$?
    fi
}

function show_conda_info() {

    rapids-logger "Check Conda info"
    conda info
    conda config --show-sources
    conda list --show-channel-urls
}
