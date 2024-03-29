# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "core")

mrc_add_pybind11_module(common SOURCE_FILES common.cpp)
mrc_add_pybind11_module(coro SOURCE_FILES coro.cpp)
mrc_add_pybind11_module(executor SOURCE_FILES executor.cpp)
mrc_add_pybind11_module(logging SOURCE_FILES logging.cpp)
mrc_add_pybind11_module(node SOURCE_FILES node.cpp)
mrc_add_pybind11_module(operators SOURCE_FILES operators.cpp)
mrc_add_pybind11_module(options SOURCE_FILES options.cpp)
mrc_add_pybind11_module(pipeline SOURCE_FILES pipeline.cpp)
mrc_add_pybind11_module(plugins SOURCE_FILES plugins.cpp)
mrc_add_pybind11_module(segment SOURCE_FILES
    segment.cpp
    segment/module_definitions/mirror_tap_orchestrator.cpp
    segment/module_definitions/segment_module_registry.cpp
    segment/module_definitions/segment_modules.cpp
)
mrc_add_pybind11_module(subscriber SOURCE_FILES subscriber.cpp)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
