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

# command used on the local system to run docker, e.g. "docker", "nvidia-docker", etc.
DOCKER_CMD=${DOCKER_CMD:-"nvidia-docker"}

# options to select which gpu devices to map into the container
DOCKER_GPU_OPTS=${DOCKER_CMD:-" "}

# extract the ci image name from the github actions definitions
CI_IMAGE=$(yq '.jobs.ci_pipe.with.test_container' < .github/workflows/pull_request.yml)

# local image name and tag
IMAGE_NAME=${IMAGE_NAME:-"mrc"}
IMAGE_TAG=${TAG:-"dev-$(date +'%y%m%d')"}

set -e

docker build --build-arg FROM_IMAGE=${CI_IMAGE} --network=host -t ${IMAGE_NAME}:${IMAGE_TAG} -f scripts/Dockerfile .

${DOCKER_CMD} run \
    ${DOCKER_GPU_OPTS} \
    --rm -d \
    -v $PWD:/work \
    -v /cores:/cores \
    -v /cache:/cache \
    --workdir /work \
    --name ${IMAGE_NAME}_${IMAGE_TAG} \
    --net host \
    --ulimit core=-1 \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_NICE \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    sleep 999999999999999999999
