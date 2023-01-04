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

function(find_and_configure_rmm version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rmm")

  include(cpm/rmm)

  # Does not work with cudf currently. Once updated to include rmm::Thrust to the GLOBAL_TARGETS. This should be used
  # rapids_cpm_rmm(
  # BUILD_EXPORT_SET    ${PROJECT_NAME}-core-exports
  # INSTALL_EXPORT_SET  ${PROJECT_NAME}-core-exports
  # )

  # Allow setting version to a variable. If so, evaluate that here. Allows for dependent versions, i.e. RMM_VERSION=${MRC_RAPIDS_VERSION}
  if("${version}" MATCHES [=[^\${(.+)}$]=])
    set(version "${${CMAKE_MATCH_1}}")
  endif()

  rapids_cpm_find(rmm ${version}
    GLOBAL_TARGETS
    rmm::rmm rmm::Thrust
    BUILD_EXPORT_SET
    ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
    ${PROJECT_NAME}-core-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/rmm.git
    GIT_TAG branch-${version}
    GIT_SHALLOW TRUE
    OPTIONS "BUILD_TESTS OFF"
    "BUILD_BENCHMARKS OFF"
    "CUDA_STATIC_RUNTIME OFF"
    "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}"
  )
endfunction()

find_and_configure_rmm(${RMM_VERSION})
