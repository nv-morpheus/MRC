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

if [[ -n "${GIT_AUTHOR_NAME}" ]] && [[ -n "${GIT_AUTHOR_EMAIL}" ]]
then
    echo "setting git config --global user.name ${GIT_AUTHOR_NAME}"
    echo "setting git config --global user.email ${GIT_AUTHOR_EMAIL}"
    git config --global user.name ${GIT_AUTHOR_NAME}
    git config --global user.email ${GIT_AUTHOR_EMAIL}
else
    echo "skipping git config setup"
    echo "set the following envs to configure git on startup: GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL"
fi
