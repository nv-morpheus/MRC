# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "coro")

set(MODULE_SOURCE_FILES)

# Add the module file
list(APPEND MODULE_SOURCE_FILES module.cpp)

# Create the python module
mrc_add_pybind11_module(coro
   INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
   SOURCE_FILES ${MODULE_SOURCE_FILES}
   LINK_TARGETS mrc::pymrc
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
