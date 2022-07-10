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

add_library(srf_protos)

add_library(${PROJECT_NAME}::srf_protos ALIAS srf_protos)

add_library(srf_architect_protos)

add_library(${PROJECT_NAME}::srf_architect_protos ALIAS srf_architect_protos)

add_library(srf_test_protos)

protobuf_generate_grpc_cpp(srf_protos
  HDRS srf_protos-headers
  PROTOS
  srf/protos/codable.proto
  srf/protos/remote_descriptor.proto
  srf/protos/tensor_meta_data.proto
)

protobuf_generate_grpc_cpp(srf_architect_protos
  GEN_GRPC
  HDRS srf_architect_protos-headers
  PROTOS
  srf/protos/architect.proto
)

protobuf_generate_grpc_cpp(srf_test_protos
  GEN_GRPC
  HDRS srf_test_protos-headers
  PROTOS
  srf/protos/test.proto
)

# Create a custom target just for the generated headers which is necessary for style checks without build
add_custom_target(
  srf_protos-headers-target
  DEPENDS ${srf_protos-headers} ${srf_architect_protos-headers} ${srf_test_protos-headers}
)

# Disable clang-tidy and IWYU for the proto target
set_target_properties(
  srf_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

set_target_properties(
  srf_architect_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

set_target_properties(
  srf_test_protos
  PROPERTIES
  CXX_CLANG_TIDY ""
  C_INCLUDE_WHAT_YOU_USE ""
  CXX_INCLUDE_WHAT_YOU_USE ""
  EXPORT_COMPILE_COMMANDS OFF
)

# Finally, add this to the style check dependencies
add_dependencies(style_checks srf_protos-headers-target)

install(
  TARGETS srf_protos srf_architect_protos
  EXPORT ${PROJECT_NAME}-core-exports
  PUBLIC_HEADER
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/protos"
)
