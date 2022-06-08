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

function(find_and_configure_prometheus_cpp version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "prometheus_cpp")

  rapids_cpm_find(prometheus-cpp ${version}
    GLOBAL_TARGETS
      prometheus-cpp prometheus-cpp::core
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp.git
      GIT_TAG "v${version}"
      GIT_SHALLOW TRUE
      PATCH_COMMAND   git checkout -- . && git apply --whitespace=fix ${PROJECT_SOURCE_DIR}/cmake/deps/patches/prometheus_export_fix.patch
      OPTIONS "BUILD_SHARED_LIBS OFF"
              "ENABLE_PULL OFF"
              "ENABLE_PUSH OFF"
              "ENABLE_COMPRESSION OFF"
              "ENABLE_TESTING OFF"
              "USE_THIRDPARTY_LIBRARIES OFF"
              "OVERRIDE_CXX_STANDARD_FLAGS OFF"
              "THIRDPARTY_CIVETWEB_WITH_SSL OFF"
              "GENERATE_PKGCONFIG OFF"
  )

endfunction()

find_and_configure_prometheus_cpp(${PROMETHEUS_CPP_VERSION})
## Manually export prometheus-core. Use this if we don't want to apply the export fix patch.
#add_library(prometheus-cpp-core STATIC IMPORTED)
#set_property(TARGET prometheus-cpp-core PROPERTY
#  IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/_deps/prometheus-cpp-build/lib/libprometheus-cpp-core.a")
