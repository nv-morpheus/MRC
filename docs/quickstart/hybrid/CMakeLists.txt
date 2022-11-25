# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "hybrid")

set(QUICKSTART_HYBRID_HOME "${CMAKE_CURRENT_SOURCE_DIR}")

create_python_package(mrc_qs_hybrid)

add_subdirectory(mrc_qs_hybrid/common)

add_subdirectory(mrc_qs_hybrid/ex00_wrap_data_objects)
add_subdirectory(mrc_qs_hybrid/ex01_wrap_nodes)

set(extra_args "IS_INPLACE")

if(MRC_PYTHON_PERFORM_INSTALL)
  list(APPEND extra_args "INSTALL_WHEEL")
endif()

if(TARGET mrc-package-install)
  list(APPEND extra_args "PYTHON_DEPENDENCIES" "mrc-package-install")
endif()

build_python_package(mrc_qs_hybrid ${extra_args})

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
