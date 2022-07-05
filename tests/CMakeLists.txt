# SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Keep all source files sorted!!!
add_executable(test_srf
  test_channel.cpp
  test_executor.cpp
  test_main.cpp
  test_metrics.cpp
  test_node.cpp
  test_pipeline.cpp
  test_segment.cpp
  test_srf.cpp
  test_type_utils.cpp
)

target_link_libraries(test_srf
  PRIVATE
    ${PROJECT_NAME}::libsrf
    GTest::gtest
)

add_test(
  NAME test_srf
  COMMAND $<TARGET_FILE:test_srf>
)

add_subdirectory(benchmarking)
add_subdirectory(logging)
