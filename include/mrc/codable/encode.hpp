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

#include "mrc/codable/api.hpp"
#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/type_traits.hpp"
#include "mrc/utils/sfinae_concept.hpp"

#include <memory>

namespace mrc::codable {

template <typename T>
class Encoder final
{
  public:
    Encoder(IEncodableStorage& storage) : m_storage(storage) {}

    void serialize(const T& obj, const EncodingOptions& opts = {})
    {
        auto parent = m_storage.push_context(typeid(T));
        detail::serialize(sfinae::full_concept{}, obj, *this, opts);
        m_storage.pop_context(parent);
    }

  protected:
    std::optional<idx_t> register_memory_view(memory::const_buffer_view view, bool force_register = false)
    {
        return m_storage.register_memory_view(std::move(view), force_register);
    }

    idx_t copy_to_eager_descriptor(memory::const_buffer_view view)
    {
        return m_storage.copy_to_eager_descriptor(std::move(view));
    }

    idx_t add_meta_data(const google::protobuf::Message& meta_data)
    {
        return m_storage.add_meta_data(meta_data);
    }

    idx_t create_memory_buffer(std::size_t bytes)
    {
        return m_storage.create_memory_buffer(bytes);
    }

    void copy_to_buffer(idx_t buffer_idx, memory::const_buffer_view view)
    {
        m_storage.copy_to_buffer(buffer_idx, std::move(view));
    }

    template <typename U>
    Encoder<U> rebind()
    {
        return Encoder<U>(m_storage);
    }

    IEncodableStorage& storage()
    {
        return m_storage;
    }

  private:
    IEncodableStorage& m_storage;

    friend T;
    friend codable_protocol<T>;
};

template <typename T>
void encode(const T& obj, IEncodableStorage& storage, EncodingOptions opts = {})
{
    Encoder<T> encoder(storage);
    encoder.serialize(obj, std::move(opts));
}

template <typename T>
void encode(const T& obj, IEncodableStorage* storage, EncodingOptions opts = {})
{
    Encoder<T> encoder(*storage);
    encoder.serialize(obj, std::move(opts));
}

}  // namespace mrc::codable
