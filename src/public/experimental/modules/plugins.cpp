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

#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace fs = boost::filesystem;

namespace srf::modules {

std::map<std::string, std::shared_ptr<PluginModule>> PluginModule::s_plugin_map{};
std::recursive_mutex PluginModule::s_mutex{};

const std::string PluginModule::PluginEntrypointList{"SRF_MODULE_entrypoint_list"};
const std::string PluginModule::PluginEntrypointLoad{"SRF_MODULE_entrypoint_load"};
const std::string PluginModule::PluginEntrypointUnload{"SRF_MODULE_entrypoint_unload"};

std::shared_ptr<PluginModule> PluginModule::create_or_acquire(const std::string& plugin_library_name)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);
    auto iter_lock = s_plugin_map.find(plugin_library_name);
    if (iter_lock != s_plugin_map.end())
    {
        return iter_lock->second;
    }

    auto plugin_ptr                   = std::shared_ptr<PluginModule>(new PluginModule(plugin_library_name));
    s_plugin_map[plugin_library_name] = plugin_ptr;

    return plugin_ptr;
}

PluginModule::PluginModule(std::string plugin_library_name) : m_plugin_library_name(std::move(plugin_library_name)) {}

void PluginModule::set_library_directory(std::string directory_path)
{
    fs::path lib_dir(directory_path);

    if (!fs::is_directory(lib_dir))
    {
        std::stringstream sstream;

        sstream << "Failed to set library directory -> '" << directory_path << "' is not a directory";
        throw std::invalid_argument(sstream.str());
    }

    m_plugin_library_dir = std::move(directory_path);
}

void PluginModule::reset_library_directory()
{
    m_plugin_library_dir = "";
}

void PluginModule::get_entrypoint(const std::string& entrypoint_name, void** entrypoint)
{
    *entrypoint = nullptr;

    dlerror();
    void* _fn = dlsym(m_plugin_handle, entrypoint_name.c_str());

    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        std::stringstream sstream;

        sstream << "Failed to find entrypoint -> '" << entrypoint_name << "' in '" << m_plugin_library_name << " : "
                << dlsym_error;

        VLOG(2) << sstream.str();
        throw std::invalid_argument(sstream.str());
    }

    *entrypoint = _fn;
}

std::vector<std::string> PluginModule::list_modules()
{
    const char** module_list;
    unsigned int module_count = m_plugin_list(&module_list);

    std::vector<std::string> ret{};
    for (int i = 0; i < module_count; i++)
    {
        ret.emplace_back(module_list[i]);
    }

    return ret;
}

bool PluginModule::load(bool throw_on_error)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (m_loaded)
    {
        return true;
    }

    if (!try_open_library_handle(throw_on_error))
    {
        return false;
    }

    if (!try_build_plugin_interface(throw_on_error))
    {
        return false;
    }

    return try_load_plugin(throw_on_error);
}

bool PluginModule::unload(bool throw_on_error)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (!m_loaded)
    {
        return true;
    }

    if (!try_unload_plugin(throw_on_error))
    {
        return false;
    }

    if (!try_close_library_handle(throw_on_error))
    {
        return false;
    }

    return true;
}

void PluginModule::reload()
{
    unload();
    load();
}

bool PluginModule::try_open_library_handle(bool throw_on_error)
{
    if (m_plugin_handle != nullptr)
    {
        return true;
    }

    std::string library_path =
        m_plugin_library_dir.empty() ? m_plugin_library_name : m_plugin_library_dir + "/" + m_plugin_library_name;

    m_plugin_handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m_plugin_handle == nullptr)
    {
        std::stringstream sstream;

        sstream << "Failed to open plugin module -> " << dlerror();
        if (throw_on_error)
        {
            LOG(ERROR) << sstream.str();
            throw std::runtime_error(sstream.str());
        }

        LOG(WARNING) << sstream.str();
        return false;
    }

    return true;
}

bool PluginModule::try_close_library_handle(bool throw_on_error)
{
    if (m_plugin_handle == nullptr)
    {
        return true;
    }

    if (dlclose(m_plugin_handle) != 0)
    {
        std::stringstream sstream;

        sstream << "Failed to close plugin module -> " << dlerror();
        if (throw_on_error)
        {
            LOG(ERROR) << sstream.str();
            throw std::runtime_error(sstream.str());
        }

        LOG(WARNING) << sstream.str();
        return false;
    }

    m_plugin_handle = nullptr;

    return true;
}

bool PluginModule::try_load_plugin(bool throw_on_error)
{
    try
    {
        m_plugin_load();
    } catch (std::exception& error)
    {
        if (throw_on_error)
        {
            LOG(ERROR) << "Plugin entrypoint load failed: " << error.what();
            throw;
        }

        LOG(WARNING) << "Plugin entrypoint load failed: " << error.what();
        return false;
    } catch (...)
    {
        if (throw_on_error)
        {
            LOG(ERROR) << "Plugin entrypoint load failed: [Unknown Exception]";
            throw;
        }

        LOG(WARNING) << "Plugin entrypoint load failed: [Unknown Exception]";
        return false;
    }

    m_loaded = true;

    return true;
}

bool PluginModule::try_unload_plugin(bool throw_on_error)
{
    try
    {
        m_plugin_unload();
        clear_plugin_interface();

        m_loaded = false;
    } catch (std::exception& error)
    {
        if (throw_on_error)
        {
            LOG(ERROR) << "Plugin entrypoint unload failed: " << error.what();
            throw;
        }

        LOG(WARNING) << "Plugin entrypoint unload failed: " << error.what();
        return false;
    } catch (...)
    {
        if (throw_on_error)
        {
            LOG(ERROR) << "Plugin entrypoint unload failed: [Unknown Exception]";
            throw;
        }

        LOG(WARNING) << "Plugin entrypoint unload failed: [Unknown Exception]";
        return false;
    }

    return true;
}

bool PluginModule::try_build_plugin_interface(bool throw_on_error)
{
    try
    {
        get_entrypoint(PluginEntrypointList, reinterpret_cast<void**>(&m_plugin_list));
        get_entrypoint(PluginEntrypointLoad, reinterpret_cast<void**>(&m_plugin_load));
        get_entrypoint(PluginEntrypointUnload, reinterpret_cast<void**>(&m_plugin_unload));
    } catch (std::invalid_argument& error)
    {
        clear_plugin_interface();
        if (throw_on_error)
        {
            LOG(ERROR) << "Failed to load SRF plugin-> " << error.what();
            throw error;
        }

        LOG(WARNING) << "Failed to load SRF plugin-> " << error.what();
        return false;
    }

    return true;
}

void PluginModule::clear_plugin_interface()
{
    m_plugin_list   = nullptr;
    m_plugin_load   = nullptr;
    m_plugin_unload = nullptr;
}

}  // namespace srf::modules
