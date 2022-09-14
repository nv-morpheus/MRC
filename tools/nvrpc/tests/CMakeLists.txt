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

include(GRPCGenerateCPP)

add_library(nvrpc-testing-protos)

protobuf_generate_grpc_cpp(
  nvrpc-testing-protos
  HDRS nvrpc-testing-protos-headers
  GEN_GRPC
  PROTOS testing.proto
)

# Create a custom target just for the generated headers which is necessary for style checks without build
add_custom_target(
  nvrpc-testing-protos-headers-target
  DEPENDS ${nvrpc-testing-protos-headers}
)

# Ignore linting for these targets
set_target_properties(nvrpc-testing-protos
  PROPERTIES
    CXX_CLANG_TIDY ""
    C_INCLUDE_WHAT_YOU_USE ""
    CXX_INCLUDE_WHAT_YOU_USE ""
    EXPORT_COMPILE_COMMANDS OFF
)

# Keep all source files sorted!!!
add_executable(test_nvrpc
  test_main.cc
  test_pingpong.cc
  test_resources.cc
  test_server.cc
)

target_link_libraries(test_nvrpc
  PRIVATE
    nvrpc
    nvrpc-client
    nvrpc-testing-protos
    GTest::gtest
    ${PROJECT_NAME}::libsrf
)

# Finally, add this to the style check dependencies
add_dependencies(${PROJECT_NAME}_style_checks nvrpc-testing-protos-headers-target)

add_test(
  NAME nvrpc
  COMMAND $<TARGET_FILE:test_nvrpc>
)
