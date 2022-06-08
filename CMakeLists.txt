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
option(SRF_BUILD_DOCS "Enable building the python bindings for Srf" OFF)
option(SRF_BUILD_LIBRARY "Whether the entire SRF library should be built. If set to OFF, only the pieces needed for a target will be built. Set to ON if installing the library" ON)
option(SRF_BUILD_PYTHON "Enable building the python bindings for Srf" ON)
option(SRF_BUILD_TESTS "Whether or not to build SRF tests" ON)
option(SRF_ENABLE_MATX "Enables the header only library MatX. Requires CUDA 11.4." OFF)
option(SRF_ENABLE_XTENSOR "Enables the xtensor library." OFF)
option(SRF_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(SRF_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" OFF)
option(SRF_USE_CONDA "Enables finding dependencies via conda instead of vcpkg. Note: This will disable vcpkg. All dependencies must be installed first in the conda environment" OFF)
option(SRF_USE_IWYU "Enable running include-what-you-use as part of the build process" OFF)

set(RAPIDS_VERSION "21.10" CACHE STRING "Which version of RAPIDs to build for. Sets default versions for RAPIDs CMake and RMM.")

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
  VERSION 22.06.00
  LANGUAGES C CXX
)

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
add_custom_target(style_checks
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
add_subdirectory(cpp)


####################################
# - Post dependencies setup --------
include(cmake/setup_compiler.cmake)

# Setup IWYU if enabled
include(cmake/setup_iwyu.cmake)

###################################################################################################
# - subdirectories --------------------------------------------------------------------------------

if (SRF_BUILD_PYTHON)
  add_subdirectory(python)
endif ()

if (SRF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif ()

add_subdirectory(tools/nvrpc)

###################################################################################################
# - install export --------------------------------------------------------------------------------

set(doc_string
[=[
Provide targets for srf.
]=])

set(code_string "")

set(rapids_project_version_compat SameMinorVersion)

rapids_export(INSTALL ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libsrf
  VERSION
  NAMESPACE srf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

###################################################################################################
# - build export ----------------------------------------------------------------------------------

rapids_export(BUILD ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libsrf
  VERSION
  LANGUAGES C CXX CUDA
  NAMESPACE srf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

if(SRF_BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Uncomment the following to print all available targets
# include(debug_utils)
# print_all_targets()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
