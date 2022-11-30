# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit REQUIRED)

include(python_module_tools)

# Create the mrc python package
create_python_package(mrc)

# Add a few additional files to be copied
add_python_sources("pytest.ini" "tests/string_reader_input.txt")

# Save the root of the python for relative paths
set(MRC_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# A common macro for adding some default arguments to add_pybind11_module
macro(mrc_add_pybind11_module)
  # Build up the common arguments for add_pybind11_module
  set(_common_args)
  list(APPEND _common_args "LINK_TARGETS" "pymrc")

  if(MRC_PYTHON_INPLACE_BUILD)
    list(APPEND _common_args "COPY_INPLACE")
  endif()

  if(MRC_PYTHON_BUILD_STUBS)
    list(APPEND _common_args "BUILD_STUBS")
  endif()

  # Forward all common arguments plus any arguments passed in
  add_pybind11_module(${ARGN} ${_common_args})
endmacro()

add_subdirectory(mrc/_pymrc)
add_subdirectory(mrc/core)
add_subdirectory(mrc/benchmarking)

# ##################################################################################################
# - tests -----------------------------------------------------------------------------------------
if(MRC_BUILD_TESTS)
  add_subdirectory(mrc/tests)
endif()

# Complete the python package
set(extra_args "")

if(MRC_PYTHON_INPLACE_BUILD)
  list(APPEND extra_args "IS_INPLACE")
endif()

if(MRC_PYTHON_PERFORM_INSTALL)
  list(APPEND extra_args "INSTALL_WHEEL")
endif()

build_python_package(mrc ${extra_args})

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
