# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(find_and_configure_morpheus_utils version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_utils")

  # TODO(Devin): Make this an ExternalProject so we don't have any rapids/rpm deps and can pull earlier.
  rapids_cpm_find(morpheus_utils ${version}
    CPM_ARGS
      #GIT_REPOSITORY https://github.com/nv-morpheus/utilities.git
      GIT_REPOSITORY /home/drobison/Development/devin-morpheus-utils-public
      GIT_TAG v${version}
      DOWNLOAD_ONLY TRUE
  )

  set(MORPHEUS_UTILS_HOME "${morpheus_utils_SOURCE_DIR}" CACHE INTERNAL "Morpheus utils home")
endfunction()

find_and_configure_morpheus_utils(${MORPHEUS_UTILS_VERSION})
