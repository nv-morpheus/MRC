/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/modules/plugins.hpp"

#include <memory>
#include <string>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class PluginProxy
{
  public:
    static std::shared_ptr<mrc::modules::PluginModule> create_or_acquire(const std::string& plugin_library_name);
};

#pragma GCC visibility pop

}  // namespace mrc::pymrc
