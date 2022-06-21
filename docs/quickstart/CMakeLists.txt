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

project(srf-quickstart
  VERSION 22.06.00
  LANGUAGES C CXX
)

# Add the Conda environment to the prefix path and add the CMake files
list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

# Setup dependencies
set(Python3_FIND_VIRTUALENV "FIRST")
set(Python3_FIND_STRATEGY "LOCATION")
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)

find_package(srf REQUIRED)

# To make it easier for CI to find output files, set the default executable suffix to .x if not set
if("${CMAKE_EXECUTABLE_SUFFIX}" STREQUAL "")
  set(CMAKE_EXECUTABLE_SUFFIX ".x")
endif()

include(python_module_tools)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(QUICKSTART_HOME "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(cpp)
add_subdirectory(hybrid)
add_subdirectory(python)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
