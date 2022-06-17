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

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

set(QUICKSTART_PYTHON_HOME "${CMAKE_CURRENT_SOURCE_DIR}")

create_python_package(srf_qs_python)

# add_subdirectory(srf_qs_python/common)

# add_subdirectory(srf_qs_python/ex00_wrap_data_objects)
# add_subdirectory(srf_qs_python/ex01_wrap_nodes)
# add_subdirectory(srf_qs_python/ex02_mixed_execution)


# # Get all of the python files to copy to the build directory
# file(GLOB_RECURSE SRF_QUICKSTART_PYTHON_PYTHON_FILES
#   LIST_DIRECTORIES FALSE
#   CONFIGURE_DEPENDS
#   "*.py"
#   "py.typed"
#   "pyproject.toml"
#   "setup.cfg"
#   "MANIFEST.in"
# )

# add_python_sources(${SRF_QUICKSTART_PYTHON_PYTHON_FILES})

build_python_package(srf_qs_python)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
