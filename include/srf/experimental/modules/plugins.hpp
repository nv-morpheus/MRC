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

class ModulePluginLibrary
{
    using module_plugin_map_t = std::map<std::string, std::mutex>;

  public:
    ModulePluginLibrary(ModulePluginLibrary&&)      = delete;
    ModulePluginLibrary(const ModulePluginLibrary&) = delete;

    ~ModulePluginLibrary() = default;

    void operator=(const ModulePluginLibrary&) = delete;

    static std::unique_ptr<ModulePluginLibrary> acquire(std::unique_ptr<ModulePluginLibrary> uptr_plugin,
                                                        std::string plugin_library_path);

    // Configuration so that dependent libraries will be searched for in
    // 'path' during OpenLibraryHandle.
    void set_library_directory(const std::string& path);

    // Reset any configuration done by SetLibraryDirectory.
    void reset_library_directory();

    /**
     * Load plugin module -- will load the plugin library and call its loader entrypoint to register
     * any modules it contains.
     */
    void load();

    /**
     * Unload the plugin module -- this will call the unload entrypoint of the plugin, which will then
     * unload any registered models.
     */
    void unload();

    /**
     * Return a list of modules published by the plugin
     */
    unsigned int list_modules(const char** list);

  private:
    explicit ModulePluginLibrary() = delete;
    explicit ModulePluginLibrary(std::string plugin_library_path) :
      m_plugin_library_path(std::move(plugin_library_path))
    {}

    static std::mutex s_mutex;
    static module_plugin_map_t s_plugin_map;

    static const std::string PluginEntrypointLoad;
    static const std::string PluginEntrypointUnload;
    static const std::string PluginEntrypointList;

    void* m_plugin_handle{nullptr};

    bool m_loaded{false};
    std::string m_plugin_library_path{};

    bool (*m_plugin_load)();
    bool (*m_plugin_unload)();
    unsigned int (*m_plugin_list)(const char**);

    void open_library_handle();
    void get_plugin_entrypoint(const std::string& entrypoint_name, void** entrypoint);
};

}  // namespace srf::modules
