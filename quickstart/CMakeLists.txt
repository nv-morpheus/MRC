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

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(srf-quickstart
  VERSION 22.04.00
  LANGUAGES C CXX
)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(srf REQUIRED)

add_library(srf_ext_quickstart SHARED
  src/sources.cpp
)

target_include_directories(srf_ext_quickstart
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(srf_ext_quickstart
  PUBLIC
    srf::libsrf
)

add_subdirectory(examples/cpp)