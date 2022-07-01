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

list(APPEND CMAKE_MESSAGE_CONTEXT "nvrpc")

# Keep all source files sorted!!!
add_library(nvrpc
  src/executor.cc
  src/server.cc
)

configure_codecov(nvrpc)

add_library(nvrpc-client
  src/client/client_executor.cc
)

configure_codecov(nvrpc-client)

add_library(${PROJECT_NAME}::nvrpc ALIAS nvrpc)
add_library(${PROJECT_NAME}::nvrpc-client ALIAS nvrpc-client)

target_link_libraries(nvrpc
  PUBLIC
    Boost::fiber
    Boost::context
    glog::glog
    gRPC::grpc++
    gRPC::grpc
    gRPC::gpr
)

target_link_libraries(nvrpc-client
  PUBLIC
    Boost::fiber
    Boost::context
    glog::glog
    gRPC::grpc++
    gRPC::grpc
    gRPC::gpr
)

target_include_directories(nvrpc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_include_directories(nvrpc-client
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(nvrpc PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_nvrpc)
set_target_properties(nvrpc-client PROPERTIES OUTPUT_NAME nvrpc-client)

install(
  TARGETS nvrpc
  EXPORT ${PROJECT_NAME}-core-exports
  RUNTIME DESTINATION  ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  TARGETS nvrpc-client
  EXPORT ${PROJECT_NAME}-core-exports
  RUNTIME DESTINATION  ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(SRF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
