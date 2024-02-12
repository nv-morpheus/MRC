#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

include_guard(GLOBAL)

function(morpheus_utils_configure_UCXX)
  list(APPEND CMAKE_MESSAGE_CONTEXT "UCXX")

  morpheus_utils_assert_cpm_initialized()
  set(UCXX_VERSION "0.37.00" CACHE STRING "Which version of UCXX to use.")

  find_package(ucx REQUIRED)

  # TODO(MDD): Switch back to the official UCXX repo once the following PR is merged:
  # https://github.com/rapidsai/ucxx/pull/166
  rapids_cpm_find(ucxx ${UCXX_VERSION}
    GLOBAL_TARGETS
      ucxx ucxx::ucxx
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY          https://github.com/pentschev/ucxx.git
      GIT_TAG                 mrc-all
      GIT_SHALLOW             TRUE
      SOURCE_SUBDIR           cpp
      OPTIONS                 "UCXX_ENABLE_RMM ON"
                              "BUILD_TESTS OFF"
                              "UCXX_ENABLE_PYTHON ON"
  )

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)

endfunction()
