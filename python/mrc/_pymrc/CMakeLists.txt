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

find_package(Python 3.8 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)
find_package(prometheus-cpp REQUIRED)

# Keep all source files sorted!!!
add_library(pymrc
  src/executor.cpp
  src/logging.cpp
  src/module_registry.cpp
  src/module_wrappers/pickle.cpp
  src/module_wrappers/shared_memory.cpp
  src/node.cpp
  src/operators.cpp
  src/options.cpp
  src/pipeline.cpp
  src/plugins.cpp
  src/py_segment_module.cpp
  src/segment.cpp
  src/segment_modules.cpp
  src/subscriber.cpp
  src/system.cpp
  src/types.cpp
  src/utilities/deserializers.cpp
  src/utilities/object_cache.cpp
  src/utilities/serializers.cpp
  src/utils.cpp
  src/watchers.cpp
)

add_library(${PROJECT_NAME}::pymrc ALIAS pymrc)

target_link_libraries(pymrc
  PUBLIC
  ${PROJECT_NAME}::libmrc
  ${Python_LIBRARIES}
  prometheus-cpp::core
  pybind11::pybind11
)

target_include_directories(pymrc
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${Python_INCLUDE_DIR}
)

set_target_properties(pymrc PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pymrc)
set_target_properties(pymrc PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Avoid doing this, it causes problems with PyInit being hidden in Cython built code.
# target_compile_options(pymrc PUBLIC "-fvisibility=hidden")
rapids_cmake_install_lib_dir(lib_dir)

install(
  TARGETS pymrc
  DESTINATION ${lib_dir}
  EXPORT ${PROJECT_NAME}-core-exports
  COMPONENT Python
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Python
)

if(MRC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
