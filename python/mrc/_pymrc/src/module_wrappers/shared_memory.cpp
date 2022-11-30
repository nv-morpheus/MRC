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

#include "pymrc/module_wrappers/shared_memory.hpp"

#include "pymrc/utilities/object_cache.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

using namespace pybind11::literals;
namespace py = pybind11;

namespace mrc::pymrc {
PythonSharedMemoryInterface::~PythonSharedMemoryInterface() = default;

PythonSharedMemoryInterface::PythonSharedMemoryInterface() : m_pycache(PythonObjectCache::get_handle())
{
    auto mod          = m_pycache.get_module("multiprocessing.shared_memory");
    m_shmem_interface = py::cast<py::object>(m_pycache.get_or_load("PythonSharedMemoryInterface.SharedMemory",
                                                                   [mod]() { return mod.attr("SharedMemory"); }));
}

void PythonSharedMemoryInterface::allocate(std::size_t sz_bytes)
{
    m_shmem = m_shmem_interface(py::none(), true, sz_bytes);
}

void PythonSharedMemoryInterface::attach(py::object block_id)
{
    m_shmem = m_shmem_interface(block_id, false);
}

void PythonSharedMemoryInterface::close()
{
    m_shmem.attr("close")();
}

py::bytes PythonSharedMemoryInterface::get_bytes() const
{
    if (m_shmem.is_none())
    {
        throw std::runtime_error("Shared memory object is none!");
    }

    return m_shmem.attr("buf").attr("tobytes")();
}

pybind11::memoryview PythonSharedMemoryInterface::get_memoryview() const
{
    if (m_shmem.is_none())
    {
        throw std::runtime_error("Shared memory object is none!");
    }

    return m_shmem.attr("buf");
}

py::object PythonSharedMemoryInterface::block_id() const
{
    if (m_shmem.is_none())
    {
        throw std::runtime_error("Shared memory object is none!");
    }

    return m_shmem.attr("name");
}

void PythonSharedMemoryInterface::free()
{
    close();
    unlink();
}

void PythonSharedMemoryInterface::set(py::bytes bytes)
{
    if (m_shmem.is_none())
    {
        throw std::runtime_error("Shared memory object is none!");
    }

    auto slice                 = py::slice(py::none(), py::none(), py::none());
    m_shmem.attr("buf")[slice] = bytes;
}

py::object PythonSharedMemoryInterface::size_bytes() const
{
    if (m_shmem.is_none())
    {
        throw std::runtime_error("Shared memory object is none!");
    }

    return m_shmem.attr("size");
}

void PythonSharedMemoryInterface::unlink()
{
    m_shmem.attr("unlink")();
}

}  // namespace mrc::pymrc

py::object mrc::pymrc::build_shmem_descriptor(const PythonSharedMemoryInterface& shmem_interface, bool flag_is_shared)
{
    auto& m_pycache = PythonObjectCache::get_handle();

    auto types            = m_pycache.get_module("types");
    auto simple_namespace = types.attr("SimpleNamespace");
    auto ns               = simple_namespace();

    py::setattr(ns, "__shared_memory_descriptor__", py::bool_(true));
    py::setattr(ns, "block_id", shmem_interface.block_id());
    py::setattr(ns, "shared", py::bool_(flag_is_shared));
    py::setattr(ns, "size_bytes", shmem_interface.size_bytes());

    return ns;
}
