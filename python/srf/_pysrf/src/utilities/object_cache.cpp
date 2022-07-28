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

#include "pysrf/utilities/object_cache.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pylifecycle.h>

#include <array>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <utility>

// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

namespace py = pybind11;
namespace srf::pysrf {

std::unique_ptr<PythonObjectCache> PythonObjectCache::s_py_object_cache{nullptr};
std::mutex PythonObjectCache::s_cache_lock{};

PythonObjectCache& PythonObjectCache::get_handle()
{
    if (!PythonObjectCache::s_py_object_cache)
    {
        std::lock_guard<std::mutex> lock(s_cache_lock);
        if (!PythonObjectCache::s_py_object_cache)
        {
            PythonObjectCache::s_py_object_cache =
                std::move(std::unique_ptr<PythonObjectCache>(new PythonObjectCache()));
        }
    }

    return *s_py_object_cache;
}

PythonObjectCache::PythonObjectCache()
{
    if (Py_IsInitialized() != 1)
    {
        std::string err = "Attempt to create PythonObjectCache before Py_Initialize()";
        LOG(ERROR) << err;
        throw std::runtime_error(err);
    }

    auto at_exit = pybind11::module_::import("atexit");
    at_exit.attr("register")(pybind11::cpp_function([this]() { this->atexit_callback(); }));
}

bool PythonObjectCache::contains(const std::string& object_id)
{
    return (m_object_cache.find(object_id) != m_object_cache.end());
}

std::size_t PythonObjectCache::size()
{
    return m_object_cache.size();
}

pybind11::object PythonObjectCache::get_or_load(const std::string& object_id, std::function<pybind11::object()> loader)
{
    std::lock_guard<std::mutex> lock(s_cache_lock);

    auto iter = m_object_cache.find(object_id);
    if (iter != m_object_cache.end())
    {
        return iter->second;
    }

    VLOG(1) << "Caching loader object: " << object_id;
    m_object_cache[object_id] = loader();  // Load the object, then drop iter to a handle.
    VLOG(1) << "Done caching loader object: " << object_id;

    return m_object_cache[object_id];
}

pybind11::object& PythonObjectCache::get_module(const std::string& module_name)
{
    std::lock_guard<std::mutex> lock(s_cache_lock);

    auto iter = m_object_cache.find(module_name);
    if (iter != m_object_cache.end())
    {
        return iter->second;
    }

    VLOG(8) << "Caching module: " << module_name;
    m_object_cache[module_name] = pybind11::module_::import(module_name.c_str());
    VLOG(8) << "Done caching module: " << module_name;
    return m_object_cache[module_name];
}

void PythonObjectCache::cache_object(const std::string& object_id, pybind11::object& obj)
{
    VLOG(8) << "Caching object: " << object_id;
    m_object_cache[object_id] = obj;
}

void PythonObjectCache::atexit_callback()
{
    py::gil_scoped_acquire gil;

    for (auto iter : m_object_cache)
    {
        VLOG(1) << "Dropping reference to cached object: '" << iter.first << "' currently has "
                << iter.second.ref_count() << " remaining references";
    }

    m_object_cache.clear();
    s_py_object_cache.release();
}

}  // namespace srf::pysrf
