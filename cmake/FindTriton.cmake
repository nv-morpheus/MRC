# SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Find Triton libraries
#
# The following variables are optionally searched for defaults
#  TRITON_ROOT_DIR: Base directory where all ibverbs components are found
#  TRITON_INCLUDE_DIR: Directory where ibverbs headers are found
#  TRITON_LIB_DIR: Directory where ibverbs libraries are found

# The following are set after configuration is done:
#  TRITON_FOUND
#  TRITON_INCLUDE_DIRS
#  TRITON_LIBRARIES

# - Find Triton
# Find the Triton library and includes
#
# TRITON_INCLUDE_DIR - where to find headers
# TRITON_LIBRARIES - List of libraries when using triton.
# TRITON_FOUND - True if libtritonserver.so found.

find_path(TRITON_INCLUDE_DIR triton/core/tritonserver.h HINTS /opt/tritonserver/include)
find_library(TRITON_LIBRARIES libtritonserver.so HINTS /opt/tritonserver/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Triton DEFAULT_MSG TRITON_LIBRARIES TRITON_INCLUDE_DIR)

if(TRITON_FOUND)
# include(CheckCXXSourceCompiles)
# CHECK_CXX_SOURCE_COMPILES("
#   #include <infiniband/verbs.h>
#   int main() {
#     struct ibv_context* ctxt;
#     struct ibv_exp_gid_attr gid_attr;
#     ibv_exp_query_gid_attr(ctxt, 1, 0, &gid_attr);
#     return 0;
#   } " HAVE_IBV_EXP)
  if(NOT TARGET Triton::libtriton)
    add_library(Triton::libtriton UNKNOWN IMPORTED)
  endif()
  set_target_properties(Triton::libtriton PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TRITON_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${TRITON_LIBRARIES}")
endif()

mark_as_advanced(
  TRITON_LIBRARIES
)
