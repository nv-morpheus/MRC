# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

# Ensure python is configured
morpheus_utils_python_configure()

morpheus_utils_print_python_info()

# Create the mrc python package
morpheus_utils_create_python_package(mrc)

# Add a few additional files to be copied
file(GLOB pymrc_test_files "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.py")
morpheus_utils_add_python_sources(
  "pytest.ini"
  "tests/string_reader_input.txt"
  ${pymrc_test_files}
)

# Build the pymrc library
add_subdirectory(mrc/_pymrc)

# Set the default link targets to avoid repeating this
morpheus_utils_python_package_set_default_link_targets(pymrc)

# Now add the python bindings
add_subdirectory(mrc/core)

# ##################################################################################################
# - Benchmarks--------------------------------------------------------------------------------------
add_subdirectory(mrc/benchmarking)

# ##################################################################################################
# - tests -----------------------------------------------------------------------------------------
if(MRC_BUILD_TESTS)
  add_subdirectory(mrc/tests)
endif()

# Complete the python package
if(MRC_PYTHON_INPLACE_BUILD)
  list(APPEND extra_args "IS_INPLACE")
endif()

if(MRC_PYTHON_BUILD_WHEEL)
  list(APPEND extra_args "BUILD_WHEEL")
endif()

if(MRC_PYTHON_PERFORM_INSTALL)
  list(APPEND extra_args "INSTALL_WHEEL")
endif()

morpheus_utils_build_python_package(mrc ${extra_args})

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
