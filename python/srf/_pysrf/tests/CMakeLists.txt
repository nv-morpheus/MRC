# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "tests")

find_package(pybind11 REQUIRED)

# Keep all source files sorted!!!
add_executable(test_pysrf
  test_executor.cpp
  test_main.cpp
  test_pipeline.cpp
  test_utils.cpp
)

target_link_libraries(test_pysrf
  PRIVATE
    ${PROJECT_NAME}::pysrf
    ${PROJECT_NAME}::libsrf
    glog::glog
    GTest::gtest
    pybind11::embed
)

add_test(
  NAME pysrf
  COMMAND $<TARGET_FILE:test_pysrf>
)

configure_codecov(test_pysrf)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
