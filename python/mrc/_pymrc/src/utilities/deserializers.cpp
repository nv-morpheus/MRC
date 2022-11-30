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

#include "pymrc/utilities/deserializers.hpp"

#include "pymrc/module_wrappers/pickle.hpp"
#include "pymrc/module_wrappers/shared_memory.hpp"

#include <glog/logging.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <ostream>

namespace py = pybind11;
namespace mrc::pymrc {

pybind11::object Deserializer::deserialize(const char* bytes, std::size_t count)
{
    VLOG(8) << "Deserializing from c++ buffer";
    auto obj = deserialize(py::bytes(bytes, count));

    // If we've unpickled a shared memory descriptor then we need to do another step to rebuild the original
    // object from shared memory.
    if (pybind11::hasattr(obj, "__shared_memory_descriptor__"))
    {
        obj = load_from_shared_memory(obj);
    }

    return obj;
}

pybind11::object Deserializer::deserialize(pybind11::bytes py_bytes)
{
    VLOG(8) << "Deserializing from py_bytes object";
    try
    {
        auto pkl = PythonPickleInterface();
        return pkl.unpickle(py_bytes);
    } catch (pybind11::error_already_set err)
    {
        LOG(ERROR) << "Failed to deserialize bytes into python object: " << err.what();
        throw;
    }
}

pybind11::object Deserializer::deserialize(pybind11::buffer_info& buffer_info)
{
    VLOG(8) << "Deserializing from py_buffer_info";
    return deserialize((const char*)buffer_info.ptr, buffer_info.size);
}

pybind11::object Deserializer::load_from_shared_memory(pybind11::object descriptor)
{
    VLOG(8) << "Deserialzing from shared memory descriptor";

    auto pkl       = PythonPickleInterface();
    auto shmem     = PythonSharedMemoryInterface();
    bool is_shared = pybind11::cast<bool>(descriptor.attr("shared"));
    pybind11::object obj;

    shmem.attach(descriptor.attr("block_id"));
    obj = pkl.unpickle(shmem.get_bytes());

    shmem.close();
    if (!is_shared)
    {
        shmem.unlink();
    }

    return obj;
}
}  // namespace mrc::pymrc
