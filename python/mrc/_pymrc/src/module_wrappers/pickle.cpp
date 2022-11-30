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

#include "pymrc/module_wrappers/pickle.hpp"

#include "pymrc/utilities/object_cache.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <array>
#include <memory>
#include <ostream>

namespace py = pybind11;
namespace mrc::pymrc {

PythonPickleInterface::~PythonPickleInterface() = default;

PythonPickleInterface::PythonPickleInterface() : m_pycache(PythonObjectCache::get_handle())
{
    auto mod = m_pycache.get_module("pickle");

    m_func_loads = m_pycache.get_or_load("PythonPickleInterface.loads", [mod]() { return mod.attr("loads"); });
    m_func_dumps = m_pycache.get_or_load("PythonPickleInterface.dumps", [mod]() { return mod.attr("dumps"); });
}

pybind11::bytes PythonPickleInterface::pickle(pybind11::object obj)
{
    try
    {
        return m_func_dumps(obj);
    } catch (pybind11::error_already_set err)
    {
        LOG(ERROR) << "Object serialization failed: " << err.what();
        throw;
    }
}

pybind11::object PythonPickleInterface::unpickle(pybind11::bytes bytes)
{
    try
    {
        return m_func_loads(bytes);
    } catch (pybind11::error_already_set err)
    {
        LOG(ERROR) << "Object deserialization failed: " << err.what();
        throw;
    }
}

}  // namespace mrc::pymrc
