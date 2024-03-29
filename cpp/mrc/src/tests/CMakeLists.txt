# SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(MRC_INTERNAL_TEST_BINARY_FILES "")

file(GLOB_RECURSE MRC_INTERNAL_TEST_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  LIST_DIRECTORIES FALSE
  CONFIGURE_DEPENDS
  "data/*.bin"
)

# For each file, copy if its different
foreach(file ${MRC_INTERNAL_TEST_SOURCE_FILES})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
    COMMENT "Copying ${file} to ${CMAKE_CURRENT_BINARY_DIR}/${file}"
  )

  list(APPEND MRC_INTERNAL_TEST_BINARY_FILES ${CMAKE_CURRENT_BINARY_DIR}/${file})
endforeach()

add_custom_target(copy_fixture_files ALL
  DEPENDS ${MRC_INTERNAL_TEST_BINARY_FILES}
)

# Keep all source files sorted!!!
add_executable(test_mrc_private
  common.cpp
  nodes/common_nodes.cpp
  pipelines/multi_segment.cpp
  pipelines/single_segment.cpp
  segments/common_segments.cpp
  test_codable.cpp
  test_control_plane_components.cpp
  test_control_plane.cpp
  test_expected.cpp
  test_grpc.cpp
  test_main.cpp
  test_memory.cpp
  test_network.cpp
  test_next.cpp
  test_partitions.cpp
  test_pipeline.cpp
  test_ranges.cpp
  test_remote_descriptor.cpp
  test_resources.cpp
  test_reusable_pool.cpp
  test_runnable.cpp
  test_service.cpp
  test_system.cpp
  test_topology.cpp
  test_ucx.cpp
)

target_link_libraries(test_mrc_private
  PRIVATE
  ${PROJECT_NAME}::libmrc
  GTest::gtest
  hwloc::hwloc
  ucx::ucs
  ucx::ucp
  pthread
  mrc_test_protos
)

# Necessary include to prevent IWYU from showing absolute paths
target_include_directories(test_mrc_private
  PRIVATE
  ${MRC_ROOT_DIR}/cpp/mrc/src
  copy_fixture_files
)

add_test(
  NAME test_mrc_private
  COMMAND $<TARGET_FILE:test_mrc_private>
)
