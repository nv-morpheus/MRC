/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/memory/adaptors.hpp"

#include <atomic>

namespace mrc::internal::memory {

template <typename UpstreamT>
class CallbackAdaptor;

/**
 * @brief Object used to collect a set of callbacks used to construct a CallbackAdaptor
 *
 * Each set of callback consists of an allocate and deallocate pair. Each are called with a pointer (ptr) to the start
 * of the memory region which was allocated or will be deallocated and the size of the memory region.
 *
 * This object simply collects a set of callback pairs and is required for the construction of a CallbackAdaptor.
 */
class CallbackBuilder
{
  public:
    virtual ~CallbackBuilder() = default;

    using allocate_callback_t   = std::function<void(void* ptr, std::size_t bytes)>;
    using deallocate_callback_t = std::function<void(void* ptr, std::size_t bytes)>;

    void register_callbacks(allocate_callback_t allocate_cb, deallocate_callback_t deallocate_cb)
    {
        CHECK(allocate_cb && deallocate_cb);
        m_allocate_callbacks.push_back(allocate_cb);
        m_deallocate_callbacks.push_back(deallocate_cb);
    }

    std::size_t size() const
    {
        return m_allocate_callbacks.size();
    }

  private:
    std::vector<allocate_callback_t> m_allocate_callbacks;
    std::vector<deallocate_callback_t> m_deallocate_callbacks;

    template <typename UpstreamT>
    friend class CallbackAdaptor;
};

/**
 * @brief A memory resource adaptor that extended a downstream resource by issues callbacks after allocation and before
 * deallocation.
 */
template <typename UpstreamT>
class CallbackAdaptor final : public mrc::memory::adaptor<UpstreamT>
{
  public:
    using allocate_callback_t   = CallbackBuilder::allocate_callback_t;
    using deallocate_callback_t = CallbackBuilder::deallocate_callback_t;

    CallbackAdaptor(UpstreamT upstream, CallbackBuilder&& builder) :
      mrc::memory::adaptor<UpstreamT>(std::move(upstream)),
      m_allocate_callbacks(std::move(builder.m_allocate_callbacks)),
      m_deallocate_callbacks(std::move(builder.m_deallocate_callbacks))
    {}

  private:
    void* do_allocate(std::size_t bytes) final
    {
        void* ptr = this->resource().allocate(bytes);

        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        for (auto& cb : m_allocate_callbacks)
        {
            cb(ptr, bytes);
        }

        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        for (auto& cb : m_deallocate_callbacks)
        {
            cb(ptr, bytes);
        }
        this->resource().deallocate(ptr, bytes);
    }

    std::mutex m_mutex;
    std::vector<allocate_callback_t> m_allocate_callbacks;
    std::vector<deallocate_callback_t> m_deallocate_callbacks;
};

}  // namespace mrc::internal::memory
