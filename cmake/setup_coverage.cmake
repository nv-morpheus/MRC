# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ######################################################################################################################
# * CMake properties ------------------------------------------------------------------------------

list(APPEND CMAKE_MESSAGE_CONTEXT "coverage")

# Include coverage tools if enabled
if(MRC_ENABLE_CODECOV)
  include(cmake/deps/Configure_gcov.cmake)

  message(STATUS "MRC_ENABLE_CODECOV is ON, configuring report exclusions and setting up coverage build targets")
  set(CODECOV_REPORT_EXCLUSIONS
    "${CMAKE_BINARY_DIR}/protos/*" # Remove this if/when we get protobuf code unit tested.
    ".cache/*"
    "cpp/mrc/benchmarks/*" # Remove this if/when we get protobuf code unit tested.
    "cpp/mrc/src/tests/*"
    "cpp/mrc/tests/*"
    "docs/*" # Remove this if/when we get protobuf code unit tested.
    "python/mrc/_pymrc/tests/*"
    "python/mrc/tests/*"
  )

  if (DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(PARALLEL_LEVEL ${CMAKE_BUILD_PARALLEL_LEVEL})
  else()
    # Get the default from the number of cores
    cmake_host_system_information(RESULT PARALLEL_LEVEL QUERY NUMBER_OF_LOGICAL_CORES)
  endif()

  # Delete the gcna files after use, and exclude dumb branches
  set(GCOVR_ADDITIONAL_ARGS "--exclude-unreachable-branches" "--exclude-throw-branches" "--delete" "-j" "${PARALLEL_LEVEL}")

  setup_target_for_coverage_gcovr_html(
    NAME gcovr-html-report-cpp
    EXCLUDE ${CODECOV_REPORT_EXCLUSIONS}
    EXECUTABLE "ctest"
    EXECUTABLE_ARGS "--exclude-regex"
    EXECUTABLE_ARGS "'test_srf_private|nvrpc'"
  )

  setup_target_for_coverage_gcovr_html(
    NAME gcovr-html-report-python
    EXCLUDE ${CODECOV_REPORT_EXCLUSIONS}
    EXECUTABLE "pytest"
    EXECUTABLE_ARGS "python"
  )

  setup_target_for_coverage_gcovr_xml(
    NAME gcovr-xml-report-cpp
    EXCLUDE ${CODECOV_REPORT_EXCLUSIONS}
    EXECUTABLE "ctest"
    EXECUTABLE_ARGS "--exclude-regex"
    EXECUTABLE_ARGS "'test_srf_private|nvrpc'"
  )

  setup_target_for_coverage_gcovr_xml(
    NAME gcovr-xml-report-python
    EXCLUDE ${CODECOV_REPORT_EXCLUSIONS}
    EXECUTABLE "pytest"
    EXECUTABLE_ARGS "python"
  )

  append_coverage_compiler_flags()
endif()

#[=======================================================================[
@brief : Given a target, configure the target with appropriate gcov if
MRC_ENABLE_CODECOV is enabled.

ex. #configure_codecov(target_name)
results --

#configure_codecov <TARGET_NAME>
#]=======================================================================]
function(configure_codecov_target target)
  if(${MRC_ENABLE_CODECOV} STREQUAL "ON")
    message(STATUS "Configuring target <${target}> for code coverage.")
    append_coverage_compiler_flags_to_target("${target}")
  endif()
endfunction()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
