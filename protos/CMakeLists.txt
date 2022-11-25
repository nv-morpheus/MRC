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

include(GRPCGenerateCPP)

add_library(mrc_protos)

add_library(${PROJECT_NAME}::mrc_protos ALIAS mrc_protos)

add_library(mrc_architect_protos)

add_library(${PROJECT_NAME}::mrc_architect_protos ALIAS mrc_architect_protos)

add_library(mrc_test_protos)

protobuf_generate_grpc_cpp(mrc_protos
  HDRS mrc_protos-headers
  PROTOS
  mrc/protos/codable.proto
  mrc/protos/remote_descriptor.proto
  mrc/protos/tensor_meta_data.proto
)

protobuf_generate_grpc_cpp(mrc_architect_protos
  GEN_GRPC
  HDRS mrc_architect_protos-headers
  PROTOS
  mrc/protos/architect.proto
)

protobuf_generate_grpc_cpp(mrc_test_protos
  GEN_GRPC
  HDRS mrc_test_protos-headers
  PROTOS
  mrc/protos/test.proto
)

# Create a custom target just for the generated headers which is necessary for style checks without build
add_custom_target(
  mrc_protos-headers-target
  DEPENDS ${mrc_protos-headers} ${mrc_architect_protos-headers} ${mrc_test_protos-headers}
)

# Disable clang-tidy and IWYU for the proto target
set_target_properties(
  mrc_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

set_target_properties(
  mrc_architect_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

set_target_properties(
  mrc_test_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

# Finally, add this to the style check dependencies
add_dependencies(${PROJECT_NAME}_style_checks mrc_protos-headers-target)

install(
  TARGETS mrc_protos mrc_architect_protos
  EXPORT ${PROJECT_NAME}-core-exports
  PUBLIC_HEADER
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/protos"
)
