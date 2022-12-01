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

# NOTE: this script should be exectued from the root of the mrc repo
# ./scripts/devel.sh

source "ci/runner/common.sh"

export DOCKER_REGISTRY_SERVER=""
export DOCKER_REGISTRY_PATH="mrc"
export DOCKER_TARGET="dev"
export DOCKER_TAG_PREFIX=""
export DOCKER_TAG_SUFFIX="$(date +'%y%m%d')"

export SKIP_PUSH=yes
SKIP_RUN=${SKIP_RUN:-""}

IMAGE_NAME=$(get_image_full_name ${DOCKER_TARGET})
CONTAINER_NAME=$(echo ${IMAGE_NAME} | sed -e 's/:/-/g')

./ci/runner/build_and_push.sh

DOCKER_CMD=${DOCKER_CMD:-"docker"}
DOCKER_GPU_OPTS=${DOCKER_GPU_OPTS:-"--gpus=all"}

if [ "${SKIP_RUN}" == "" ]; then
  ${DOCKER_CMD} run \
    ${DOCKER_GPU_OPTS} \
    --rm -d \
    -v $PWD:/work \
    -v /cores:/cores \
    -v /cache:/cache \
    --workdir /work \
    --name ${CONTAINER_NAME} \
    --net host \
    --ulimit core=-1 \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_NICE \
    ${IMAGE_NAME} \
    sleep 999999999999999999999
fi
