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

if(MRC_USE_IWYU)
  morpheus_utils_initialize_iwyu(
      MRC_USE_IWYU
      MRC_IWYU_VERBOSITY
      MRC_IWYU_PROGRAM
      MRC_IWYU_OPTIONS
      MRC_USE_CCACHE
  )
endif(MRC_USE_IWYU)