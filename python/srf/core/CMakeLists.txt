# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

srf_add_pybind11_module(common            SOURCE_FILES common.cpp)
srf_add_pybind11_module(executor          SOURCE_FILES executor.cpp)
srf_add_pybind11_module(logging           SOURCE_FILES logging.cpp)
srf_add_pybind11_module(node              SOURCE_FILES node.cpp)
srf_add_pybind11_module(operators         SOURCE_FILES operators.cpp)
srf_add_pybind11_module(options           SOURCE_FILES options.cpp)
srf_add_pybind11_module(pipeline          SOURCE_FILES pipeline.cpp)
srf_add_pybind11_module(plugins           SOURCE_FILES plugins.cpp)
srf_add_pybind11_module(segment_modules   SOURCE_FILES segment_modules.cpp)
srf_add_pybind11_module(segment           SOURCE_FILES segment.cpp)
srf_add_pybind11_module(subscriber        SOURCE_FILES subscriber.cpp)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
