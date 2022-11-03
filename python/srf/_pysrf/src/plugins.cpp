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

#include "pysrf/plugins.hpp"

#include <memory>

namespace srf::pysrf {

std::shared_ptr<srf::modules::PluginModule> PluginProxy::create_or_acquire(const std::string& plugin_library_name)
{
    return modules::PluginModule::create_or_acquire(plugin_library_name);
}

std::vector<std::string> PluginProxy::list_modules(srf::modules::PluginModule& self)
{
    return self.list_modules();
}

bool load(srf::modules::PluginModule& self, bool throw_on_error)
{
    return self.load(throw_on_error);
}

bool unload(srf::modules::PluginModule& self, bool throw_on_error)
{
    return self.unload(throw_on_error);
}

void reload(srf::modules::PluginModule& self)
{
    return self.reload();
}

void reset_library_directory(srf::modules::PluginModule& self)
{
    return self.reset_library_directory();
}

void set_library_directory(srf::modules::PluginModule& self, std::string path)
{
    return self.set_library_directory(path);
}

}  // namespace srf::pysrf
