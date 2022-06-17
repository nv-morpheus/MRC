# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# srf_add_pybind11_module(NAME common SOURCE common.cpp)
# srf_add_pybind11_module(NAME executor SOURCE executor.cpp)
# srf_add_pybind11_module(NAME logging SOURCE logging.cpp)
# srf_add_pybind11_module(NAME node SOURCE node.cpp)
# srf_add_pybind11_module(NAME operators SOURCE operators.cpp)
# srf_add_pybind11_module(NAME options SOURCE options.cpp)
# srf_add_pybind11_module(NAME pipeline SOURCE pipeline.cpp)
# srf_add_pybind11_module(NAME segment SOURCE segment.cpp)
# srf_add_pybind11_module(NAME segment_definition SOURCE segment_definition.cpp)
# srf_add_pybind11_module(NAME subscriber SOURCE subscriber.cpp)

add_pybind11_module(common SOURCE_FILES common.cpp LINK_TARGETS pysrf OUTPUT_TARGET common_target)
add_pybind11_module(executor SOURCE_FILES executor.cpp LINK_TARGETS pysrf OUTPUT_TARGET executor_target)
add_pybind11_module(logging SOURCE_FILES logging.cpp LINK_TARGETS pysrf OUTPUT_TARGET logging_target)
add_pybind11_module(node SOURCE_FILES node.cpp LINK_TARGETS pysrf OUTPUT_TARGET node_target)
add_pybind11_module(operators SOURCE_FILES operators.cpp LINK_TARGETS pysrf OUTPUT_TARGET operators_target)
add_pybind11_module(options SOURCE_FILES options.cpp LINK_TARGETS pysrf OUTPUT_TARGET options_target)
add_pybind11_module(pipeline SOURCE_FILES pipeline.cpp LINK_TARGETS pysrf OUTPUT_TARGET pipeline_target)
add_pybind11_module(segment SOURCE_FILES segment.cpp LINK_TARGETS pysrf OUTPUT_TARGET segment_target)
# add_pybind11_module(segment_definition SOURCE_FILES segment_definition.cpp LINK_TARGETS pysrf OUTPUT_TARGET segment_definition_target)
add_pybind11_module(subscriber SOURCE_FILES subscriber.cpp LINK_TARGETS pysrf OUTPUT_TARGET subscriber_target)

if(SRF_PYTHON_INPLACE_BUILD)
  inplace_build_copy(${common_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${executor_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${logging_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${node_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${operators_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${options_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${pipeline_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${segment_target} ${CMAKE_CURRENT_SOURCE_DIR})
  # inplace_build_copy(${segment_definition_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${subscriber_target} ${CMAKE_CURRENT_SOURCE_DIR})
endif()
