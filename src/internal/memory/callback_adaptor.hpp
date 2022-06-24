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

#include <atomic>
#include <srf/memory/adaptors.hpp>

namespace srf::internal::memory {

template <typename UpstreamT>
class CallbackAdaptor : public srf::memory::adaptor<UpstreamT>
{
  public:
    using allocate_callback_t   = std::function<void(void* ptr, std::size_t bytes)>;
    using deallocate_callback_t = std::function<void(void* ptr, std::size_t bytes)>;

    CallbackAdaptor(UpstreamT upstream, std::size_t callback_slots) :
      srf::memory::adaptor<UpstreamT>(std::move(upstream)),
      m_callback_slots(callback_slots)
    {}

    void register_callbacks(allocate_callback_t allocate_cb, deallocate_callback_t deallocate_cb)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        CHECK_LT(m_allocate_callbacks.size(), m_callback_slots);
        CHECK_LT(m_deallocate_callbacks.size(), m_callback_slots);
        m_allocate_callbacks.push_back(allocate_cb);
        m_deallocate_callbacks.push_back(deallocate_cb);
    }

  private:
    void* do_allocate(std::size_t bytes) final
    {
        void* ptr = this->upstream().allocate(bytes);

        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        DCHECK_EQ(m_allocate_callbacks.size(), m_callback_slots);
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
        this->upstream().deallocate(ptr, bytes);
    }

    std::mutex m_mutex;
    std::size_t m_callback_slots;
    std::vector<allocate_callback_t> m_allocate_callbacks;
    std::vector<deallocate_callback_t> m_deallocate_callbacks;
};

}  // namespace srf::internal::memory
