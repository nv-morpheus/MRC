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

list(APPEND CMAKE_MESSAGE_CONTEXT "quickstart")

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Add the Conda environment to the prefix path and add the CMake files
list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

# Add the RAPIDS cmake helper scripts
include(import_rapids_cmake)

project(mrc-quickstart
  VERSION 22.11.00
  LANGUAGES C CXX
)

# Ensure CPM is initialized
rapids_cpm_init()

# Set the option prefix to match the outer project before including. Must be before find_package(mrc)
set(OPTION_PREFIX "MRC")
include(python_module_tools)

rapids_find_package(mrc REQUIRED)
rapids_find_package(CUDAToolkit REQUIRED)

# To make it easier for CI to find output files, set the default executable suffix to .x if not set
if("${CMAKE_EXECUTABLE_SUFFIX}" STREQUAL "")
  set(CMAKE_EXECUTABLE_SUFFIX ".x")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(QUICKSTART_HOME "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(cpp)
add_subdirectory(hybrid)
add_subdirectory(python)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
