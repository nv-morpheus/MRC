# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "mrc")

# Global options.
# Options of the form: MRC_BUILD_XXX, enable library features and use default values targeting average users of the library
# Options of the form: MRC_ENABLE_XXX, alter whether to build using a specific optional third party library. ON by default
# Options of the form: MRC_USE_XXX, enable linting or alter the environment and are OFF by default
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(MRC_BUILD_BENCHMARKS "Whether or not to build MRC benchmarks" OFF)
option(MRC_BUILD_DOCS "Enable building of API documentation" OFF)
option(MRC_BUILD_LIBRARY "Whether the entire MRC library should be built.
  If set to OFF, only the pieces needed for a target will be built. Set to ON if installing the library" ON)
option(MRC_BUILD_PYTHON "Enable building the python bindings for MRC" ON)
option(MRC_BUILD_TESTS "Whether or not to build MRC tests" ON)
option(MRC_ENABLE_CODECOV "Enable gcov code coverage" OFF)
option(MRC_ENABLE_DEBUG_INFO "Enable printing debug information" OFF)
option(MRC_PYTHON_INPLACE_BUILD "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)
option(MRC_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(MRC_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" OFF)
option(MRC_USE_IWYU "Enable running include-what-you-use as part of the build process" OFF)

set(MRC_RAPIDS_VERSION "25.02" CACHE STRING "Which version of RAPIDS to build for. Sets default versions for RAPIDS CMake and RMM.")

set(MRC_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MRC_CACHE_DIR)

enable_testing()

if (MRC_USE_IWYU AND MRC_USE_CCACHE)
  message(FATAL_ERROR "MRC_USE_IWYU and MRC_USE_CCACHE cannot be set simultaneously")
endif()

# MRC CMake path and module extensions
set(MRC_CMAKE_MODULE_PATH_EXTENSIONS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/utilities/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/utilities/cmake/morpheus_utils/package_search"
    )

set(MRC_CMAKE_PREFIX_PATH_EXTENSIONS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    )

# Prepend path and prefix updates so they take priority in this scope.
list(PREPEND CMAKE_MODULE_PATH "${MRC_CMAKE_MODULE_PATH_EXTENSIONS}")
list(PREPEND CMAKE_PREFIX_PATH "${MRC_CMAKE_PREFIX_PATH_EXTENSIONS}")

# Load morpheus utils and update CMake paths
set(MORPHEUS_UTILS_RAPIDS_VERSION ${MRC_RAPIDS_VERSION})
set(MORPHEUS_UTILS_RAPIDS_CPM_INIT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapids_cpm_package_overrides.json")
include(morpheus_utils/load)

# Configure project package manager
morpheus_utils_initialize_package_manager()

# Configure CUDA architecture
# NOTE: This MUST occur before any 'project' calls because of rapids_cmake requirements.
morpheus_utils_initialize_cuda_arch(mrc)

project(mrc
  VERSION 25.10.00
  LANGUAGES C CXX
)

morpheus_utils_initialize_install_prefix()

# Delay enabling CUDA until after we have determined our CXX compiler
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  message(STATUS "Setting CUDA host compiler to match CXX compiler: ${CMAKE_CXX_COMPILER}")

  # Only set the host compiler if we arent using clang. Using clang > 8ish is
  # incompatible with CUDA 11.4/11.5/11.6. See Issue #102
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Our version of NVCC officially only supports clang versions 3.2 - 13, we are now using 14
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

  # Check if the major version of Clang is greater than 15
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "--version"
      OUTPUT_VARIABLE clang_version_info
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "version [0-9]+" clang_version_value ${clang_version_info})
  string(REGEX REPLACE "version ([0-9]+)" "\\1" clang_version ${clang_version_value})

  if(${clang_version} VERSION_LESS_EQUAL "15")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines-ts")
  endif()

  # Suppress unused argument during compilation due to -L being included
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
endif()

# Now enable CUDA
enable_language(CUDA)

# Create a variable for subdirectories to use to reference from the root of the
# project. Also used by ccache
set(MRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Setup cache before dependencies
# Configure CCache if requested
include(environment/init_ccache)

# Disable exporting compile commands for dependencies
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# Create a custom target to allow preparing for style checks
add_custom_target(${PROJECT_NAME}_style_checks
  COMMENT "Building dependencies for style checks"
)

# Configure all dependencies
include(dependencies)

# Enable for all first party code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# To make it easier for CI to find output files, set the default executable suffix to .x if not set
if("${CMAKE_EXECUTABLE_SUFFIX}" STREQUAL "")
  set(CMAKE_EXECUTABLE_SUFFIX ".x")
endif()

add_subdirectory(protos)

# ###################################
# - Post dependencies setup --------
morpheus_utils_compiler_set_defaults(MRC_USE_CLANG_TIDY)

# Setup code coverage components
include(environment/init_coverage)

# Setup IWYU if enabled
include(environment/init_iwyu)

# ##################################################################################################
# - subdirectories ---------------------------------------------------------------------------------

add_subdirectory(cpp)

if(MRC_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(MRC_BUILD_DOCS)
  add_subdirectory(docs)
endif()

# ##################################################################################################
# - install export ---------------------------------------------------------------------------------

set(doc_string
  [=[
Provide targets for mrc.
]=])

set(code_string "")

set(rapids_project_version_compat SameMinorVersion)

# Need to explicitly set VERSION ${PROJECT_VERSION} here since rapids_cmake gets
# confused with the `RAPIDS_VERSION` variable we use
rapids_export(INSTALL ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-exports
  GLOBAL_TARGETS libmrc pymrc
  COMPONENTS python
  COMPONENTS_EXPORT_SET ${PROJECT_NAME}-python-exports
  VERSION ${PROJECT_VERSION}
  NAMESPACE mrc::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# - build export -----------------------------------------------------------------------------------
rapids_export(BUILD ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-exports
  GLOBAL_TARGETS libmrc pymrc
  COMPONENTS python
  COMPONENTS_EXPORT_SET ${PROJECT_NAME}-python-exports
  VERSION ${PROJECT_VERSION}
  LANGUAGES C CXX CUDA
  NAMESPACE mrc::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# - debug info -------------------------------------------------------------------------------------
if (MRC_ENABLE_DEBUG_INFO)
  morpheus_utils_print_all_targets()

  morpheus_utils_print_target_properties(
      TARGETS
        libmrc
        pymrc
      WRITE_TO_FILE
  )

  morpheus_utils_print_global_properties(
      WRITE_TO_FILE
  )
endif()

# Cleanup the environment after we exit this scope
list(REMOVE_ITEM CMAKE_PREFIX_PATH "${MRC_CMAKE_PREFIX_PATH_EXTENSIONS}")
list(REMOVE_ITEM CMAKE_MODULE_PATH "${MRC_CMAKE_MODULE_PATH_EXTENSIONS}")

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
