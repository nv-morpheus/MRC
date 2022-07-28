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

#include <pybind11/pytypes.h>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace srf::pysrf {

#pragma GCC visibility push(default)
/**
 * @brief Cache python objects in a way that allows them to be freed correctly before the interpreter shuts down.
 **/
class PythonObjectCache
{
  public:
    /**
     * @brief Get singleton handle
     * @return Reference to singleton PythonObjectCache
     */
    static PythonObjectCache& get_handle();

    ~PythonObjectCache() = default;

    PythonObjectCache(PythonObjectCache&)  = delete;
    PythonObjectCache(PythonObjectCache&&) = delete;

    PythonObjectCache& operator=(PythonObjectCache&)  = delete;
    PythonObjectCache& operator=(PythonObjectCache&&) = delete;

    /**
     *
     * @return Number of objects in the cache -- approximate
     */
    std::size_t size();

    /**
     * @brief Returns true/false if 'object_id' is in the cache.
     * @param object_id
     * @return Object is cached (true) or not (false)
     */
    bool contains(const std::string& object_id);

    /**
     *
     * @param object_id String tag to cache the object under.
     * @param loader Arbitrary loader function used to get the pybind object to cache. Will be called to
     * populate the cache entry if 'object_id' doesn't exist.
     * @return Cached object
     */
    pybind11::object get_or_load(const std::string& object_id, std::function<pybind11::object()> loader);

    /**
     * @brief Shortcut to obtain / cache a module object.
     * @param module_name Module name, with either be retrieved or loaded, cached and retrieved.
     * @return Loaded module as pybind object.
     */
    pybind11::object& get_module(const std::string& module_name);

    /**
     * @brief Add an arbitrary python object to the cache. If an entry with the same name exists it will be
     * overwritten.
     */
    void cache_object(const std::string& object_id, pybind11::object& obj);

  private:
    static std::mutex s_cache_lock;
    static std::unique_ptr<PythonObjectCache> s_py_object_cache;

    PythonObjectCache();

    /**
     * @brief Actions taken to clear the cache prior to interpreter shutdown.
     */
    void atexit_callback();

    std::map<std::string, pybind11::object> m_object_cache;
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
