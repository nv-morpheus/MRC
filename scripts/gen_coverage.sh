#!/usr/bin/env bash
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -x
set -e

cmake -DCMAKE_BUILD_TYPE=Debug -GNinja -DCMAKE_BUILD_TYPE=Debug -DMRC_ENABLE_CODECOV=ON -DMRC_BUILD_PYTHON=ON -DMRC_BUILD_TESTS=ON -B ./build
cmake --build ./build
pip install -e ./build/python
cd ./build && ctest && pytest ./build/python/tests
cmake --build ./build --target gcovr-html-report
