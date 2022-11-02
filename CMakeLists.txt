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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "srf")

# Global options.
# Options of the form: SRF_BUILD_XXX, enable library features and use default values targeting average users of the library
# Options of the form: SRF_ENABLE_XXX, alter whether to build using a specific optional third party library. ON by default
# Options of the form: SRF_USE_XXX, enable linting or alter the environment and are OFF by default
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(SRF_BUILD_BENCHMARKS "Whether or not to build SRF benchmarks" OFF)
option(SRF_BUILD_DOCS "Enable building of API documentation" OFF)
option(SRF_BUILD_LIBRARY "Whether the entire SRF library should be built. If set to OFF, only the pieces needed for a target will be built. Set to ON if installing the library" ON)
option(SRF_BUILD_PYTHON "Enable building the python bindings for Srf" ON)
option(SRF_BUILD_TESTS "Whether or not to build SRF tests" ON)
option(SRF_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(SRF_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" OFF)
option(SRF_USE_CONDA "Enables finding dependencies via conda instead of vcpkg. Note: This will disable vcpkg. All dependencies must be installed first in the conda environment" ON)
option(SRF_USE_IWYU "Enable running include-what-you-use as part of the build process" OFF)
option(SRF_ENABLE_CODECOV "Enable gcov code coverage" OFF)

set(SRF_RAPIDS_VERSION "22.08" CACHE STRING "Which version of RAPIDS to build for. Sets default versions for RAPIDS CMake and RMM.")

set(SRF_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(SRF_CACHE_DIR)

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/setup_package_manager.cmake)

enable_testing()

# Add the RAPIDS cmake helper scripts
include(cmake/import_rapids_cmake.cmake)

# Default to using "NATIVE" for CUDA_ARCHITECTURES to build based on GPU in system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES was not defined. Defaulting to '${CMAKE_CUDA_ARCHITECTURES}' to build only for local architecture. Specify -DCMAKE_CUDA_ARCHITECTURES='ALL' to build for all archs.")
endif()

rapids_cuda_init_architectures(srf)

project(srf
  VERSION 22.11.00
  LANGUAGES C CXX
)

rapids_cmake_write_version_file(${CMAKE_CURRENT_SOURCE_DIR}/include/srf/version.hpp)

# Delay enabling CUDA until after we have determined our CXX compiler
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  message(STATUS "Setting CUDA host compiler to match CXX compiler: ${CMAKE_CXX_COMPILER}")

  # Only set the host compiler if we arent using clang. Using clang > 8ish is
  # incompatible with CUDA 11.4/11.5/11.6. See Issue #102
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
endif()

# Now enable CUDA
enable_language(CUDA)

# Create a variable for subdirectories to use to reference from the root of the
# project. Also used by ccache
set(SRF_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Once the build type is set, remove any dumb vcpkg debug folders from the
# search paths. Without this FindBoost fails since it defaults to the debug
# binaries
if(DEFINED CACHE{SRF_VCPKG_TOOLCHAIN} AND DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
  message(STATUS "Release Build: Removing debug Vcpkg paths from CMAKE_PREFIX_PATH and CMAKE_FIND_ROOT_PATH")
  list(REMOVE_ITEM CMAKE_PREFIX_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
  list(REMOVE_ITEM CMAKE_FIND_ROOT_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Setup cache before dependencies
include(cmake/setup_cache.cmake)

# Configure conda
if(SRF_USE_CONDA AND DEFINED ENV{CONDA_PREFIX})
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE STRING "" FORCE)
  endif()
endif()

# Disable exporting compile commands for dependencies
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# Create a custom target to allow preparing for style checks
add_custom_target(${PROJECT_NAME}_style_checks
  COMMENT "Building dependencies for style checks"
)

# Configure all dependencies
include(cmake/dependencies.cmake)

# Enable for all first party code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# To make it easier for CI to find output files, set the default executable suffix to .x if not set
if("${CMAKE_EXECUTABLE_SUFFIX}" STREQUAL "")
  set(CMAKE_EXECUTABLE_SUFFIX ".x")
endif()

add_subdirectory(protos)

####################################
# - Post dependencies setup --------
include(cmake/setup_compiler.cmake)

# Setup code coverage components
include(cmake/setup_coverage.cmake)

# Setup IWYU if enabled
include(cmake/setup_iwyu.cmake)

# ###################################
# - Begin SRF Targets --------------

# Keep all source files sorted!!!
add_library(libsrf
  src/internal/data_plane/callbacks.cpp
  src/internal/data_plane/client.cpp
  src/internal/data_plane/request.cpp
  src/internal/data_plane/resources.cpp
  src/internal/data_plane/server.cpp
  src/internal/executor/executor.cpp
  src/internal/executor/iexecutor.cpp
  src/internal/memory/device_resources.cpp
  src/internal/memory/host_resources.cpp
  src/internal/memory/transient_pool.cpp
  src/internal/network/resources.cpp
  src/internal/pipeline/controller.cpp
  src/internal/pipeline/instance.cpp
  src/internal/pipeline/ipipeline.cpp
  src/internal/pipeline/manager.cpp
  src/internal/pipeline/pipeline.cpp
  src/internal/pipeline/port_graph.cpp
  src/internal/pipeline/resources.cpp
  src/internal/resources/manager.cpp
  src/internal/resources/partition_resources_base.cpp
  src/internal/resources/partition_resources.cpp
  src/internal/runnable/engine.cpp
  src/internal/runnable/engine_factory.cpp
  src/internal/runnable/engines.cpp
  src/internal/runnable/fiber_engine.cpp
  src/internal/runnable/fiber_engines.cpp
  src/internal/runnable/resources.cpp
  src/internal/runnable/thread_engine.cpp
  src/internal/runnable/thread_engines.cpp
  src/internal/segment/builder.cpp
  src/internal/segment/definition.cpp
  src/internal/segment/ibuilder.cpp
  src/internal/segment/idefinition.cpp
  src/internal/segment/instance.cpp
  src/internal/service.cpp
  src/internal/system/device_info.cpp
  src/internal/system/device_partition.cpp
  src/internal/system/engine_factory_cpu_sets.cpp
  src/internal/system/fiber_manager.cpp
  src/internal/system/fiber_pool.cpp
  src/internal/system/fiber_task_queue.cpp
  src/internal/system/gpu_info.cpp
  src/internal/system/host_partition.cpp
  src/internal/system/host_partition_provider.cpp
  src/internal/system/iresources.cpp
  src/internal/system/isystem.cpp
  src/internal/system/partition.cpp
  src/internal/system/partition_provider.cpp
  src/internal/system/partitions.cpp
  src/internal/system/resources.cpp
  src/internal/system/system.cpp
  src/internal/system/system_provider.cpp
  src/internal/system/thread.cpp
  src/internal/system/thread.cpp
  src/internal/system/thread_pool.cpp
  src/internal/system/topology.cpp
  src/internal/ucx/context.cpp
  src/internal/ucx/endpoint.cpp
  src/internal/ucx/memory_block.cpp
  src/internal/ucx/receive_manager.cpp
  src/internal/ucx/resources.cpp
  src/internal/ucx/worker.cpp
  src/internal/utils/collision_detector.cpp
  src/internal/utils/exception_guard.cpp
  src/internal/utils/parse_config.cpp
  src/internal/utils/parse_ints.cpp
  src/internal/utils/shared_resource_bit_map.cpp
  src/public/benchmarking/tracer.cpp
  src/public/benchmarking/trace_statistics.cpp
  src/public/benchmarking/util.cpp
  src/public/channel/channel.cpp
  src/public/codable/encoded_object.cpp
  src/public/core/addresses.cpp
  src/public/core/bitmap.cpp
  src/public/core/executor.cpp
  src/public/core/fiber_pool.cpp
  src/public/core/logging.cpp
  src/public/cuda/device_guard.cpp
  src/public/cuda/sync.cpp
  src/public/experimental/modules/module_registry.cpp
  src/public/experimental/modules/plugins.cpp
  src/public/experimental/modules/sample_modules.cpp
  src/public/experimental/modules/segment_modules.cpp
  src/public/manifold/manifold.cpp
  src/public/memory/buffer_view.cpp
  src/public/metrics/counter.cpp
  src/public/metrics/registry.cpp
  src/public/node/edge_adapter_registry.cpp
  src/public/node/edge_builder.cpp
  src/public/node/edge_registry.cpp
  src/public/node/port_registry.cpp
  src/public/options/engine_groups.cpp
  src/public/options/fiber_pool.cpp
  src/public/options/options.cpp
  src/public/options/placement.cpp
  src/public/options/resources.cpp
  src/public/options/services.cpp
  src/public/options/topology.cpp
  src/public/runnable/context.cpp
  src/public/runnable/launcher.cpp
  src/public/runnable/runnable.cpp
  src/public/runnable/runner.cpp
  src/public/runnable/types.cpp
  src/public/segment/builder.cpp
  src/public/segment/definition.cpp
  src/public/utils/bytes_to_string.cpp
  src/public/utils/thread_utils.cpp
  src/public/utils/type_utils.cpp

  # todo(#63) - Enable UCX Context/Workers and UCX Registered Memory Resources
  # src/internal/data_plane/client_worker.cpp
  # src/internal/data_plane/client.cpp
  # src/internal/data_plane/instance.cpp
  # src/internal/resources/host_resources.cpp
  # src/internal/resources/partition_resources.cpp
  # src/internal/resources/resource_partitions.cpp
  # src/internal/resources/system_resources.cpp
)

add_library(${PROJECT_NAME}::libsrf ALIAS libsrf)

target_link_libraries(libsrf
  PUBLIC
  srf_protos
  srf_architect_protos
  rmm::rmm
  CUDA::nvml
  CUDA::cudart
  rxcpp::rxcpp
  glog::glog
  libcudacxx::libcudacxx
  PRIVATE
  nvrpc
  nvrpc-client
  hwloc::hwloc
  prometheus-cpp::core # private in MR !199
  ucx::ucs
  ucx::ucp
)

target_include_directories(libsrf
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(libsrf
  PUBLIC
  $<$<BOOL:${SRF_BUILD_BENCHMARKS}>:SRF_ENABLE_BENCHMARKING>
)

set_target_properties(libsrf PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

# Finally, set the install RPATH to include the stubs folder for CUDA::nvml. If thats made private, this can be removed
set_target_properties(libsrf PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib/stubs")

# ##################################################################################################
# - install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

install(
  TARGETS libsrf
  DESTINATION ${lib_dir}
  EXPORT ${PROJECT_NAME}-core-exports
  COMPONENT Core
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Core
)

# ##################################################################################################
# - subdirectories --------------------------------------------------------------------------------
if(SRF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(SRF_BUILD_TESTS)
  add_subdirectory(tests)

  add_subdirectory(src/tests)
endif()

if(SRF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

add_subdirectory(tools/nvrpc)

# ##################################################################################################
# - install export --------------------------------------------------------------------------------
set(doc_string
  [=[
Provide targets for srf.
]=])

set(code_string "")

set(rapids_project_version_compat SameMinorVersion)

# Need to explicitly set VERSION ${PROJECT_VERSION} here since rapids_cmake gets
# confused with the `RAPIDS_VERSION` variable we use
rapids_export(INSTALL ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libsrf
  VERSION ${PROJECT_VERSION}
  NAMESPACE srf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# - build export ----------------------------------------------------------------------------------
rapids_export(BUILD ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libsrf
  VERSION ${PROJECT_VERSION}
  LANGUAGES C CXX CUDA
  NAMESPACE srf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

if(SRF_BUILD_DOCS)
  add_subdirectory(docs)
endif()

add_subdirectory(src/tools)

# Uncomment the following to print all available targets
# include(debug_utils)
# print_all_targets()
list(POP_BACK CMAKE_MESSAGE_CONTEXT)
