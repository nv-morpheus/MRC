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

# Need to start by sourcing the host environment
BUILD_DIR="build-conda"

echo "Installing Python components"

# Install the python library
pushd ${SRC_DIR}/${BUILD_DIR}/python

echo "PYTHON: ${PYTHON}"
echo "which python: $(which python)"
${PYTHON} -m pip install -vv --no-deps .

popd
