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
find_package(pybind11 REQUIRED)
find_package(prometheus-cpp REQUIRED)

add_library(pysrf
  src/executor.cpp
  src/logging.cpp
  src/options.cpp
  src/pipeline.cpp
  src/segment.cpp
  src/system.cpp
  src/node.cpp
  src/operators.cpp
  src/subscriber.cpp
  src/types.cpp
  src/utils.cpp
  src/watchers.cpp
)

add_library(${PROJECT_NAME}::pysrf ALIAS pysrf)

target_link_libraries(pysrf
  PUBLIC
    ${PROJECT_NAME}::libsrf
    pybind11::pybind11
    prometheus-cpp::core
  )

target_include_directories(pysrf
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

set_target_properties(pysrf PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pysrf)
set_target_properties(pysrf PROPERTIES CXX_VISIBILITY_PRESET hidden)
# Avoid doing this, it causes problems with PyInit being hidden in Cython built code.
#   target_compile_options(pysrf PUBLIC "-fvisibility=hidden")

rapids_cmake_install_lib_dir(lib_dir)

install(
  TARGETS pysrf
  DESTINATION ${lib_dir}
  EXPORT ${PROJECT_NAME}-core-exports
  COMPONENT Python
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Python
)

if(SRF_BUILD_TESTS)
  add_subdirectory(tests)
endif()
