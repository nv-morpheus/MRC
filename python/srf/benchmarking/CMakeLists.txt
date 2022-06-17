# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
# srf_add_pybind11_module(NAME watchers SOURCE watchers.cpp)
# srf_add_pybind11_module(NAME tracers SOURCE tracer.cpp trace_statistics.cpp)

add_pybind11_module(watchers SOURCE_FILES watchers.cpp LINK_TARGETS pysrf OUTPUT_TARGET watchers_target)
add_pybind11_module(tracers SOURCE_FILES tracer.cpp trace_statistics.cpp LINK_TARGETS pysrf OUTPUT_TARGET tracers_target)

if(SRF_PYTHON_INPLACE_BUILD)
  inplace_build_copy(${watchers_target} ${CMAKE_CURRENT_SOURCE_DIR})
  inplace_build_copy(${tracers_target} ${CMAKE_CURRENT_SOURCE_DIR})
endif()
