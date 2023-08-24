# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "control-plane")

# find_package(NodeJs REQUIRED)

# # Run NPM install to ensure all libraries are installed
# add_custom_command(
#   OUTPUT  ${CMAKE_CURRENT_SOURCE_DIR}/package-lock.json
#   COMMAND ${NodeJs_NPM_EXECUTABLE} install
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package.json
#   COMMENT "Installing Node packages for Control Plane"
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
# )


# # Run tsc to build the server output
# add_custom_command(
#   OUTPUT  ${CMAKE_CURRENT_SOURCE_DIR}/dist/tsconfig.tsbuildinfo
#   COMMAND ${NodeJs_NPM_EXECUTABLE} run build:server
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package-lock.json
#   COMMENT "Running Typescript Compilation"
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
# )

# file(GLOB  compiled_js_files "${CMAKE_CURRENT_SOURCE_DIR}/dist/*.js")

# add_custom_target(control-plane ALL
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dist/tsconfig.tsbuildinfo
# )

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
