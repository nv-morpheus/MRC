# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# =============================================================================

function(configure_iwyu)
  list(APPEND CMAKE_MESSAGE_CONTEXT "iwyu")

  set(MRC_IWYU_VERBOSITY "1" CACHE STRING "Set verbosity level for include-what-you-use, 1 is default, 1 only shows recomendations and 11+ prints everything")

  find_program(MRC_IWYU_PROGRAM "include-what-you-use")

  if(MRC_IWYU_PROGRAM)
    set(MRC_IWYU_OPTIONS
      -Xiwyu; --mapping_file=${PROJECT_SOURCE_DIR}/ci/iwyu/mappings.imp;
      -Xiwyu; --max_line_length=120;
      -Xiwyu; --verbose=${MRC_IWYU_VERBOSITY};
      -Xiwyu; --no_fwd_decls;
      -Xiwyu; --quoted_includes_first;
      -Xiwyu; --cxx17ns;
      -Xiwyu --no_comments)

    # Convert these to space separated arguments
    string(REPLACE ";" " " MRC_IWYU_OPTIONS "${MRC_IWYU_OPTIONS}")

    message(STATUS "Enabling include-what-you-use for MRC targets")

    set(IWYU_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/run_iwyu.sh")

    # Make a ccache runner file with the necessary settings. MRC_CCACHE_OPTIONS must be set!
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_iwyu.sh.in" "${IWYU_WRAPPER}")

    if(MRC_USE_CCACHE)
      set(CMAKE_C_INCLUDE_WHAT_YOU_USE "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_prefix.sh;${IWYU_WRAPPER};${CMAKE_C_COMPILER}" PARENT_SCOPE)
      set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_prefix.sh;${IWYU_WRAPPER};${CMAKE_CXX_COMPILER}" PARENT_SCOPE)
    else()
      set(CMAKE_C_INCLUDE_WHAT_YOU_USE "${IWYU_WRAPPER}" PARENT_SCOPE)
      set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_WRAPPER}" PARENT_SCOPE)
    endif()

  else()
    message(WARNING "IWYU option MRC_USE_IWYU is enabled but the include-what-you-use was not found. Check iwyu installation and add the iwyu bin dir to your PATH variable.")
  endif(MRC_IWYU_PROGRAM)
endfunction()

# Configure IWYU if requested
if(MRC_USE_IWYU)
  configure_iwyu()
endif(MRC_USE_IWYU)
