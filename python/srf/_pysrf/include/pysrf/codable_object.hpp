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

#include <pysrf/types.hpp>
#include <pysrf/utilities/deserializers.hpp>
#include <pysrf/utilities/serializers.hpp>

#include <srf/codable/codable_protocol.hpp>
#include <srf/codable/encoded_object.hpp>
#include <srf/codable/encoding_options.hpp>
#include <srf/memory/block.hpp>
#include <srf/memory/memory_kind.hpp>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <glog/logging.h>

#include <iomanip>
#include <type_traits>
#include <typeindex>

namespace srf::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, pybind11::object>>>
{
    static void serialize(const T& py_object, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        using namespace srf::pysrf;
        VLOG(8) << "Serializing python object";
        pybind11::gil_scoped_acquire gil;
        pybind11::buffer_info py_bytebuffer;
        std::tuple<char*, std::size_t> serialized_obj;

        auto guard = encoded.acquire_encoding_context();

        // Serialize the object
        serialized_obj = Serializer::serialize(py_object, opts.use_shm(), !opts.force_copy());

        // Copy it or not.
        encoded.add_memory_block(memory::const_block(
            std::get<0>(serialized_obj), std::get<1>(serialized_obj), memory::memory_kind_type::host));
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        using namespace srf::pysrf;
        VLOG(8) << "De-serializing python object";
        pybind11::gil_scoped_acquire gil;
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));

        auto idx           = encoded.start_idx_for_object(object_idx);
        const auto& buffer = encoded.memory_block(idx);
        const char* data   = static_cast<const char*>(buffer.data());

        return Deserializer::deserialize(data, buffer.bytes());
    }
};

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, pysrf::PyHolder>>>
{
    static void serialize(const T& pyholder_object, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        using namespace srf::pysrf;
        VLOG(8) << "Serializing PyHolder object";
        pybind11::gil_scoped_acquire gil;
        pybind11::object py_object = pyholder_object.copy_obj();  // Not a deep copy, just inc_ref the pointer.
        pybind11::buffer_info py_bytebuffer;
        std::tuple<char*, std::size_t> serialized_obj;

        auto guard = encoded.acquire_encoding_context();

        // Serialize the object
        serialized_obj = Serializer::serialize(py_object, opts.use_shm(), !opts.force_copy());

        // Copy it or not.
        encoded.add_memory_block(memory::const_block(
            std::get<0>(serialized_obj), std::get<1>(serialized_obj), memory::memory_kind_type::host));
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        using namespace srf::pysrf;
        VLOG(8) << "De-serializing PyHolder object";
        pybind11::gil_scoped_acquire gil;
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));

        auto idx           = encoded.start_idx_for_object(object_idx);
        const auto& buffer = encoded.memory_block(idx);
        const char* data   = static_cast<const char*>(buffer.data());

        return Deserializer::deserialize(data, buffer.bytes());
    }
};

}  // namespace srf::codable
