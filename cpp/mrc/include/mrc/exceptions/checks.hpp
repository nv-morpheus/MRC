/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

namespace mrc::exceptions {

void throw_failed_check_exception(const std::string& file,
                                  const std::string& function,
                                  unsigned int line,
                                  const std::string& msg = "");

#define MRC_CHECK_THROW(condition)                                                                \
    for (std::stringstream ss; !(condition);                                                      \
         ::mrc::exceptions::throw_failed_check_exception(__FILE__, __func__, __LINE__, ss.str())) \
    ss

}  // namespace mrc::exceptions
