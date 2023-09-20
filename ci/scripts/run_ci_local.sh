#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

case "$1" in
    "" )
        STAGES=("bash")
        ;;
    "all" )
        STAGES=("checks" "build-clang" "build-codecov" "build-gcc" "test-clang" "test-codecov" "test-gcc" "docs" "benchmark" "conda")
        ;;
    "build" )
        STAGES=("build-clang" "build-codecov" "build-gcc")
        ;;
    "test" )
        STAGES=("test-clang" "test-codecov" "test-gcc")
        ;;
    "checks" | "build-clang" | "build-codecov" | "build-gcc" | "codecov" | "test" | "test-clang" | "test-codecov" | "test-gcc" | "docs" | "benchmark" | "conda" | "bash" )
        STAGES=("$1")
        ;;
    * )
        echo "Error: Invalid argument \"$1\" provided. Expected values: \"all\", \"checks\", \"build\", " \
             "\"build-clang\", \"build-codecov\", \"build-gcc\", \"codecov\", \"test\", \"test-clang\", " \
             "\"test-codecov\", \"test-gcc\", \"docs\", \"benchmark\", \"conda\" or \"bash\""
        exit 1
        ;;
esac

# CI image doesn't contain ssh, need to use https
function git_ssh_to_https()
{
    local url=$1
    echo $url | sed -e 's|^git@github\.com:|https://github.com/|'
}

MRC_ROOT=${MRC_ROOT:-$(git rev-parse --show-toplevel)}

GIT_URL=$(git remote get-url origin)
GIT_URL=$(git_ssh_to_https ${GIT_URL})

GIT_UPSTREAM_URL=$(git remote get-url upstream)
GIT_UPSTREAM_URL=$(git_ssh_to_https ${GIT_UPSTREAM_URL})

GIT_BRANCH=$(git branch --show-current)
GIT_COMMIT=$(git log -n 1 --pretty=format:%H)

LOCAL_CI_TMP=${LOCAL_CI_TMP:-${MRC_ROOT}/.tmp/local_ci_tmp}
CONTAINER_VER=${CONTAINER_VER:-230711}
CUDA_VER=${CUDA_VER:-11.8}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

BUILD_CONTAINER="nvcr.io/ea-nvidia-morpheus/morpheus:mrc-ci-build-${CONTAINER_VER}"
TEST_CONTAINER="nvcr.io/ea-nvidia-morpheus/morpheus:mrc-ci-test-${CONTAINER_VER}"

ENV_LIST="--env LOCAL_CI_TMP=/ci_tmp"
ENV_LIST="${ENV_LIST} --env GIT_URL=${GIT_URL}"
ENV_LIST="${ENV_LIST} --env GIT_UPSTREAM_URL=${GIT_UPSTREAM_URL}"
ENV_LIST="${ENV_LIST} --env GIT_BRANCH=${GIT_BRANCH}"
ENV_LIST="${ENV_LIST} --env GIT_COMMIT=${GIT_COMMIT}"
ENV_LIST="${ENV_LIST} --env PARALLEL_LEVEL=$(nproc)"
ENV_LIST="${ENV_LIST} --env CUDA_VER=${CUDA_VER}"
ENV_LIST="${ENV_LIST} --env SKIP_CONDA_ENV_UPDATE=${SKIP_CONDA_ENV_UPDATE}"

mkdir -p ${LOCAL_CI_TMP}
cp ${MRC_ROOT}/ci/scripts/bootstrap_local_ci.sh ${LOCAL_CI_TMP}

for STAGE in "${STAGES[@]}"; do
    if [[ "${STAGE}" =~ "clang" || ]]; then
        ENV_LIST="${ENV_LIST} --env BUILD_CC=clang"
    elif [[ "${STAGE}" =~ "gcc" ]]; then
        ENV_LIST="${ENV_LIST} --env BUILD_CC=gcc"
    elif [[ "${STAGE}" =~ "codecov" ]]; then
        ENV_LIST="${ENV_LIST} --env BUILD_CC=cc-coverage"
    fi

    DOCKER_RUN_ARGS="--rm -ti --net=host -v "${LOCAL_CI_TMP}":/ci_tmp ${ENV_LIST} --env STAGE=${STAGE}"
    if [[ "${STAGE}" =~ "test" || "${STAGE}" =~ "codecov" || "${USE_GPU}" == "1" ]]; then
        CONTAINER="${TEST_CONTAINER}"
        DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --runtime=nvidia --gpus all --cap-add=sys_nice --cap-add=sys_ptrace"
    else
        CONTAINER="${BUILD_CONTAINER}"
        DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --runtime=runc"
    fi

    if [[ "${STAGE}" == "bash" ]]; then
        DOCKER_RUN_CMD="bash --init-file /ci_tmp/bootstrap_local_ci.sh"
    else
        DOCKER_RUN_CMD="/ci_tmp/bootstrap_local_ci.sh"
    fi

    echo "Running ${STAGE} stage in ${CONTAINER}"
    docker run ${DOCKER_RUN_ARGS} ${DOCKER_EXTRA_ARGS} ${CONTAINER} ${DOCKER_RUN_CMD}

    STATUS=$?
    if [[ ${STATUS} -ne 0 ]]; then
        echo "Error: docker exited with a non-zero status code for ${STAGE} of ${STATUS}"
        exit ${STATUS}
    fi
done