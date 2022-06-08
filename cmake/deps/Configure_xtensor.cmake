#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_xtensor VERSION)

  list(APPEND CMAKE_MESSAGE_CONTEXT "xtensor")

  rapids_cpm_find(xtensor ${XTENSOR_VERSION}
    GLOBAL_TARGETS
      xtensor xtensor::xtensor
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/xtensor-stack/xtensor.git
      GIT_TAG         ${XTENSOR_VERSION}
      GIT_SHALLOW     TRUE
      OPTIONS         "BUILD_TESTS OFF"
                      "BUILD_BENCHMARKS OFF"
                      "DOWNLOAD_GTEST OFF"
                      "DOWNLOAD_GBENCHMARK OFF"
                      "CPP17 ON"
  )

endfunction()

find_and_configure_xtensor(${XTENSOR_VERSION})
