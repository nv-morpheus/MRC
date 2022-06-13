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

gpuci_logger "Env Setup"
source /opt/conda/etc/profile.d/conda.sh
export SRF_ROOT=${SRF_ROOT:-$(git rev-parse --show-toplevel)}
gpuci_logger "Procs: $(nproc)"
gpuci_logger "Memory"

/usr/bin/free -g

gpuci_logger "user info"
id

export CMAKE_BUILD_ALL_FEATURES="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DSRF_BUILD_BENCHMARKS=ON -DSRF_BUILD_EXAMPLES=ON -DSRF_BUILD_PYTHON=ON -DSRF_BUILD_TESTS=ON -DSRF_USE_CONDA=ON"
export CMAKE_CLANG_OPTIONS="-DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-12 -DCMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc"

# Set the depth to allow git describe to work
export GIT_DEPTH=1000

gpuci_logger "Retrieving base branch from GitHub API"
# For PRs, $GIT_BRANCH is like: pull-request/989
REPO_NAME=$(basename "${GIT_URL}" .git)
ORG_NAME=$(basename "$(dirname "${GIT_URL}")")
PR_NUM="${GIT_BRANCH##*/}"
[[ -n "$GH_TOKEN" ]] && CURL_HEADERS=('-H' "Authorization: token ${GH_TOKEN}")
RESP=$(
curl -s \
    -H "Accept: application/vnd.github.v3+json" \
    "${CURL_HEADERS[@]}" \
    "https://api.github.com/repos/${ORG_NAME}/${REPO_NAME}/pulls/${PR_NUM}"
)

BASE_BRANCH=$(echo "${RESP}" | jq -r '.base.ref')

# Change target is the branch name we are merging into but due to the weird way jenkins does
# the checkout it isn't recognized by git without the origin/ prefix
export CHANGE_TARGET="origin/${BASE_BRANCH}"
gpuci_logger "Base branch: ${BASE_BRANCH}"

# S3 vars
export S3_URL="s3://rapids-downloads/ci/srf"
export DISPLAY_URL="https://downloads.rapids.ai/ci/srf"
export ARTIFACT_URL="${S3_URL}/pull-request/${CHANGE_ID}/${GIT_COMMIT}/${NVARCH}"
export DISPLAY_ARTIFACT_URL="${DISPLAY_URL}/pull-request/${CHANGE_ID}/${GIT_COMMIT}/${NVARCH}/"

# Set sccache env vars
export SCCACHE_S3_KEY_PREFIX=srf-${NVARCH}
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768
#export SCCACHE_LOG=debug

gpuci_logger "Environ:"
env | sort
