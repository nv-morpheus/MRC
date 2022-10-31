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

#include <dlfcn.h>
#include <glog/logging.h>

#include <map>
#include <memory>
#include <mutex>
#include <sstream>

namespace srf::modules {

class PluginModule
{
    using module_plugin_map_t = std::map<std::string, std::shared_ptr<PluginModule>>;

  public:
    PluginModule()                    = delete;
    PluginModule(PluginModule&&)      = delete;
    PluginModule(const PluginModule&) = delete;

    ~PluginModule() = default;

    void operator=(const PluginModule&) = delete;

    /**
     * Prevent duplicate versions of a plugin library from existing
     * @param plugin_library_name Path to the library file
     * @return A shared pointer to an existing or newly created PluginModule
     */
    static std::shared_ptr<PluginModule> create_or_acquire(const std::string& plugin_library_name);

    // Configuration so that dependent libraries will be searched for in
    // 'path' during OpenLibraryHandle.
    void set_library_directory(std::string path);

    // Reset any configuration done by SetLibraryDirectory.
    void reset_library_directory();

    /**
     * Load plugin module -- will load the plugin library and call its loader entrypoint to register
     * any modules it contains.
     * @param throw_on_error Flag indicating if failure to load a library is an error; true by default.
     * @return true if the library was successfully loaded, false if throw_on_error is false and load failed
     */
    bool load(bool throw_on_error = true);

    /**
     * Unload the plugin module -- this will call the unload entrypoint of the plugin, which will then
     * unload any registered models.
     * @param throw_on_error Flag indicating if failure to load a library is an error; true by default.
     * @return true if the library was successfully unloaded, false if throw_on_error is false and unload failed
     */
    bool unload(bool throw_on_error = true);

    /**
     * Unload and re-load the given module
     */
    void reload();

    /**
     * Return a list of modules published by the plugin
     */
    std::vector<std::string> list_modules();

  private:
    explicit PluginModule(std::string plugin_library_name);

    static std::recursive_mutex s_mutex;
    static module_plugin_map_t s_plugin_map;

    static const std::string PluginEntrypointLoad;
    static const std::string PluginEntrypointUnload;
    static const std::string PluginEntrypointList;

    const std::string m_plugin_library_name;
    std::string m_plugin_library_dir{};

    void* m_plugin_handle{nullptr};
    bool m_loaded{false};

    bool (*m_plugin_load)();
    bool (*m_plugin_unload)();
    unsigned int (*m_plugin_list)(const char***);

    bool try_load_plugin(bool throw_on_error = true);
    bool try_unload_plugin(bool throw_on_error = true);
    bool try_build_plugin_interface(bool throw_on_error = true);
    bool try_open_library_handle(bool throw_on_error = true);
    bool try_close_library_handle(bool throw_on_error = true);

    void get_entrypoint(const std::string& entrypoint_name, void** entrypoint);
    void clear_plugin_interface();
};

}  // namespace srf::modules
