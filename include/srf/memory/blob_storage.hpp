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

#include <srf/memory/buffer.hpp>
#include <srf/memory/memory_kind.hpp>
#include <srf/utils/macros.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>

namespace srf::memory {

class IBlobStorage  // NOLINT
{
  public:
    virtual ~IBlobStorage() = default;

    inline void* data()
    {
        return do_data();
    }

    inline const void* data() const
    {
        return do_data();
    }

    inline std::size_t bytes() const
    {
        return do_bytes();
    }

    inline memory_kind_type kind() const
    {
        return do_kind();
    }

    /**
     * @brief Allocate a new storage object.
     *
     * If the underlying storage can access a memory resource to allocate a new segment of memory, this capabability is
     * passed thru and a new generic storage object is returned; otherwise an exception will be thrown.
     *
     * @param bytes
     * @param stream
     * @return std::shared_ptr<IBlobStorage>
     */
    std::shared_ptr<IBlobStorage> allocate(std::size_t bytes, cudaStream_t stream = nullptr) const
    {
        return do_allocate(bytes, stream);
    }

  private:
    virtual void* do_data()                  = 0;
    virtual const void* do_data() const      = 0;
    virtual std::size_t do_bytes() const     = 0;
    virtual memory_kind_type do_kind() const = 0;

    virtual std::shared_ptr<IBlobStorage> do_allocate(std::size_t bytes, cudaStream_t stream) const = 0;
};

template <typename StorageT>
class BlobStorage;

template <typename... Properties>  // NOLINT
class BlobStorage<buffer<Properties...>> final : public IBlobStorage
{
  public:
    BlobStorage(buffer<Properties...>&& buffer) : m_buffer(std::move(buffer)) {}
    ~BlobStorage() final = default;

  private:
    void* do_data() final
    {
        return m_buffer.data();
    }

    const void* do_data() const final
    {
        return m_buffer.data();
    }

    std::size_t do_bytes() const final
    {
        return m_buffer.bytes();
    }

    memory_kind_type do_kind() const final
    {
        return m_buffer.kind();
    }

    std::shared_ptr<IBlobStorage> do_allocate(std::size_t bytes, cudaStream_t stream) const final
    {
        CHECK(stream == nullptr);
        auto b = buffer<Properties...>(bytes, m_buffer.view());
        return std::make_shared<BlobStorage<buffer<Properties...>>>(std::move(b));
    }

    buffer<Properties...> m_buffer;
};

template <typename... Properties>  // NOLINT
class BlobStorage<std::shared_ptr<buffer<Properties...>>> final : public IBlobStorage
{
  public:
    explicit BlobStorage(std::shared_ptr<buffer<Properties...>> buffer) : m_buffer(std::move(buffer)) {}
    ~BlobStorage() final = default;

  private:
    void* do_data() final
    {
        return m_buffer->data();
    }

    const void* do_data() const final
    {
        return m_buffer->data();
    }

    std::size_t do_bytes() const final
    {
        return m_buffer->bytes();
    }

    memory_kind_type do_kind() const final
    {
        return m_buffer->kind();
    }

    std::shared_ptr<IBlobStorage> do_allocate(std::size_t bytes, cudaStream_t stream) const final
    {
        CHECK(stream == nullptr);
        auto b = buffer<Properties...>(bytes, m_buffer.view());
        return std::make_shared<BlobStorage<buffer<Properties...>>>(std::move(b));
    }

    std::shared_ptr<buffer<Properties...>> m_buffer;
};

template <>
class BlobStorage<rmm::device_buffer> final : public IBlobStorage
{
  public:
    BlobStorage(rmm::device_buffer&& buffer) : m_buffer(std::move(buffer)) {}
    ~BlobStorage() final = default;

  private:
    void* do_data() final
    {
        return m_buffer.data();
    }

    const void* do_data() const final
    {
        return m_buffer.data();
    }

    std::size_t do_bytes() const final
    {
        return m_buffer.size();
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::device;
    }

    std::shared_ptr<IBlobStorage> do_allocate(std::size_t bytes, cudaStream_t stream) const final
    {
        auto view   = rmm::cuda_stream_view(stream);
        auto buffer = rmm::device_buffer(bytes, view, m_buffer.memory_resource());
        return std::make_shared<BlobStorage<rmm::device_buffer>>(std::move(buffer));
    }

    rmm::device_buffer m_buffer;
};

template <>
class BlobStorage<std::shared_ptr<rmm::device_buffer>> final : public IBlobStorage
{
  public:
    explicit BlobStorage(std::shared_ptr<rmm::device_buffer> buffer) : m_buffer(std::move(buffer)) {}
    ~BlobStorage() final = default;

  private:
    void* do_data() final
    {
        return m_buffer->data();
    }

    const void* do_data() const final
    {
        return m_buffer->data();
    }

    std::size_t do_bytes() const final
    {
        return m_buffer->size();
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::device;
    }

    std::shared_ptr<IBlobStorage> do_allocate(std::size_t bytes, cudaStream_t stream) const final
    {
        auto view   = rmm::cuda_stream_view(stream);
        auto buffer = rmm::device_buffer(bytes, view, m_buffer->memory_resource());
        return std::make_shared<BlobStorage<rmm::device_buffer>>(std::move(buffer));
    }

    std::shared_ptr<rmm::device_buffer> m_buffer;
};

}  // namespace srf::memory
