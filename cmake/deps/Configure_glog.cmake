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

function(find_and_configure_glog version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "glog")

  rapids_cpm_find(glog ${version}
    GLOBAL_TARGETS
      glog glog::glog
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY          https://github.com/google/glog.git
      GIT_TAG                 v${version}
      GIT_SHALLOW             TRUE
      OPTIONS                 "WITH_CUSTOM_PREFIX ON"
                              "WITH_PKGCONFIG OFF"
                              "BUILD_TESTING OFF"
  )

endfunction()

find_and_configure_glog(${GLOG_VERSION})
