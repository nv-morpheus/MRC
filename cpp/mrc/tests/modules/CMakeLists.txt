# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "tests")

add_library(dynamic_test_module MODULE
  dynamic_module.cpp
)

include_directories(dynamic_test_module
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(dynamic_test_module
  PUBLIC
  rxcpp::rxcpp
  PRIVATE
  libmrc
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
