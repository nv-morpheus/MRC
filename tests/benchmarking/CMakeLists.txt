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
add_executable(test_mrc_benchmarking
  test_benchmarking.cpp
  test_main.cpp
  test_stat_gather.cpp
  test_utils.cpp
)

target_link_libraries(test_mrc_benchmarking
  PRIVATE
  ${PROJECT_NAME}::libmrc
  GTest::gtest
  prometheus-cpp::core
)

add_test(
  NAME test_mrc_benchmarking
  COMMAND $<TARGET_FILE:test_mrc_benchmarking>
)
