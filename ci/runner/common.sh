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

# required -> ${var:-"default"}
# optional -> $([ ! -z "${var+set} "] && echo ${var} || echo "default")
#             if var is unset, then default is echoed;
#             otherwise the value of var (possibly empty) is echoed

DOCKER_REGISTRY_PATH=${DOCKER_REGISTRY_PATH:-"morpheus"}
DOCKER_REGISTRY_SERVER=$([ ! -z "${DOCKER_REGISTRY_SERVER+set}" ] && echo "${DOCKER_REGISTRY_SERVER}" || echo "nvcr.io")
DOCKER_TAG_PREFIX=$([ ! -z "${DOCKER_TAG_PREFIX+set}" ] && echo "${DOCKER_TAG_PREFIX}" || echo "mrc-ci")
DOCKER_TAG_SUFFIX=$([ ! -z "${DOCKER_TAG_SUFFIX+set}" ] && echo "${DOCKER_TAG_SUFFIX}" || echo "$(date +'%y%m%d')")

# if this function receives two inputs $1 and $2 which are both set and both not empty, it will concat the two
# otherwise, the empty string is returned
function concat_if_both() {
    if [[ -n "${1+set}" ]] && [[ -n "${2+set}" ]]; then
      echo "${1}${2}"
    fi
}

function get_image_full_name() {
    if [ -z ${1} ]; then
        echo "fatal: get_image_full_name requires a positional arguement which is set and not empty"
        exit 911
    fi
    echo "$(concat_if_both ${DOCKER_REGISTRY_SERVER} "/")${DOCKER_REGISTRY_PATH}:$(concat_if_both ${DOCKER_TAG_PREFIX} "-")${1}$(concat_if_both "-" ${DOCKER_TAG_SUFFIX})"
}
