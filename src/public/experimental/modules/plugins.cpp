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

#include "srf/experimental/modules/plugins.hpp"

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace srf::modules {

std::mutex ModulePluginLibrary::s_mutex{};

const std::string ModulePluginLibrary::PluginEntrypointList{"SRF_MODULE_entrypoint_list"};
const std::string ModulePluginLibrary::PluginEntrypointLoad{"SRF_MODULE_entrypoint_load"};
const std::string ModulePluginLibrary::PluginEntrypointUnload{"SRF_MODULE_entrypoint_unload"};

std::unique_ptr<ModulePluginLibrary> ModulePluginLibrary::Acquire(std::unique_ptr<ModulePluginLibrary> uptr_plugin,
                                                                  std::string plugin_library_path)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    uptr_plugin.reset(new ModulePluginLibrary(std::move(plugin_library_path)));

    return std::move(uptr_plugin);
}

void ModulePluginLibrary::set_library_directory(const std::string& path)
{
    throw std::runtime_error("Unimplemented");
}

void ModulePluginLibrary::reset_library_directory()
{
    throw std::runtime_error("Unimplemented");
}

unsigned int ModulePluginLibrary::list_modules(const char** list)
{
    return m_plugin_list(list);
}

void ModulePluginLibrary::load()
{
    if (m_loaded)
    {
        return;
    }

    open_library_handle();
    get_plugin_entrypoint(PluginEntrypointList, reinterpret_cast<void**>(&m_plugin_list));
    get_plugin_entrypoint(PluginEntrypointLoad, reinterpret_cast<void**>(&m_plugin_load));
    get_plugin_entrypoint(PluginEntrypointUnload, reinterpret_cast<void**>(&m_plugin_unload));

    m_plugin_load();
    m_loaded = true;
}

void ModulePluginLibrary::unload()
{
    if (!m_loaded)
    {
        return;
    }

    m_plugin_unload();

    if (dlclose(m_plugin_handle) != 0)
    {
        std::stringstream sstream;

        sstream << "Failed to close plugin module -> " << dlerror();
        VLOG(2) << sstream.str();
        throw std::runtime_error(dlerror());
    }

    m_plugin_load   = nullptr;
    m_plugin_unload = nullptr;

    m_loaded = false;
}

void ModulePluginLibrary::open_library_handle()
{
    m_plugin_handle = dlopen(m_plugin_library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m_plugin_handle == nullptr)
    {
        std::stringstream sstream;

        sstream << "Failed to open plugin module -> " << dlerror();
        VLOG(2) << sstream.str();

        throw std::runtime_error(sstream.str());
    }
}

void ModulePluginLibrary::get_plugin_entrypoint(const std::string& entrypoint_name, void** entrypoint)
{
    *entrypoint = nullptr;

    dlerror();
    void* _fn = dlsym(m_plugin_handle, entrypoint_name.c_str());

    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        std::stringstream sstream;

        sstream << "Failed to find entrypoint -> '" << entrypoint_name << "' in '" << m_plugin_library_path << " : "
                << dlsym_error;

        VLOG(2) << sstream.str();
        throw std::invalid_argument(sstream.str());
    }

    *entrypoint = _fn;
}

}  // namespace srf::modules
