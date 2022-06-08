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

# Find the ibverbs libraries
#
# The following variables are optionally searched for defaults
#  IBVERBS_ROOT_DIR: Base directory where all ibverbs components are found
#  IBVERBS_INCLUDE_DIR: Directory where ibverbs headers are found
#  IBVERBS_LIB_DIR: Directory where ibverbs libraries are found

# The following are set after configuration is done:
#  IBVERBS_FOUND
#  IBVERBS_INCLUDE_DIRS
#  IBVERBS_LIBRARIES

# - Find rdma verbs
# Find the rdma verbs library and includes
#
# VERBS_INCLUDE_DIR - where to find ibverbs.h, etc.
# VERBS_LIBRARIES - List of libraries when using ibverbs.
# VERBS_FOUND - True if ibverbs found.
# HAVE_IBV_EXP - True if experimental verbs is enabled.

find_path(VERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(VERBS_LIBRARIES ibverbs HINTS /usr/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(verbs DEFAULT_MSG VERBS_LIBRARIES VERBS_INCLUDE_DIR)

if(VERBS_FOUND)
  include(CheckCXXSourceCompiles)
  CHECK_CXX_SOURCE_COMPILES("
    #include <infiniband/verbs.h>
    int main() {
      struct ibv_context* ctxt;
      struct ibv_exp_gid_attr gid_attr;
      ibv_exp_query_gid_attr(ctxt, 1, 0, &gid_attr);
      return 0;
    } " HAVE_IBV_EXP)
  if(NOT TARGET IBVerbs::verbs)
    add_library(IBVerbs::verbs UNKNOWN IMPORTED)
  endif()
  set_target_properties(IBVerbs::verbs PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${VERBS_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${VERBS_LIBRARIES}")
endif()

mark_as_advanced(
  VERBS_LIBRARIES
)
