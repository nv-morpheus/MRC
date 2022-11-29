# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_pybind11_library(
  data
  # MODULE_ROOT
  # ${QUICKSTART_HYBRID_HOME}
  SOURCE_FILES
  data.cpp
  LINK_TARGETS
  mrc::pymrc
  OUTPUT_TARGET
  common_data_target
)

target_include_directories(${common_data_target}
  PUBLIC
  ./include
)

inplace_build_copy(${common_data_target} ${CMAKE_CURRENT_SOURCE_DIR})

add_pybind11_library(
  nodes
  SOURCE_FILES
  nodes.cpp
  LINK_TARGETS
  ${common_data_target}
  OUTPUT_TARGET
  nodes_data_target
)

inplace_build_copy(${nodes_data_target} ${CMAKE_CURRENT_SOURCE_DIR})

# Set this variable in the parent scope so other examples can link to it
set(common_data_target ${common_data_target} PARENT_SCOPE)
set(nodes_data_target ${nodes_data_target} PARENT_SCOPE)
