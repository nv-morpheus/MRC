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

#include "internal/memory/resources.hpp"

#include "internal/memory/ucx_registered_resource.hpp"

#include <srf/memory/resources/arena_resource.hpp>

namespace srf::internal::memory {

Resources::Resources(resources::PartitionResourceBase& base, std::optional<ucx::Resources>& ucx_resources) :
  resources::PartitionResourceBase(base)
{
    std::unique_ptr<srf::memory::memory_resource> host_mr;
    std::unique_ptr<srf::memory::memory_resource> device_mr;

    if (partition().has_device())
    {
        DVLOG(10) << "constructing pinned and device memory resources for partition_id: " << partition_id()
                  << "; cuda_device_id: " << partition().device().cuda_device_id();
        m_host_raw   = std::make_unique<srf::memory::pinned_memory_resource>();
        m_device_raw = std::make_unique<srf::memory::cuda_malloc_resource>(partition().device().cuda_device_id());

        device_mr = std::make_unique<srf::memory::callback_memory_resource>(
            device_alloc_cb, device_dealloc_cb, m_device_raw->kind());
    }
    else
    {
        m_host_raw = std::make_unique<srf::memory::malloc_memory_resource>();
    }

    srf::memory::allocate_callback_t host_alloc_cb = [this, &ucx_resources](std::size_t bytes,
                                                                            void* user_data) -> void* {
        void* ptr = m_host_raw->allocate(bytes);

        if (ptr == nullptr)
        {
            throw std::bad_alloc{};
        }

        VLOG(5) << "device " << partition().device().cuda_device_id() << " allocation:" << ptr
                << "; size=" << bytes_to_string(bytes);

        if (ucx_resources)
        {
            auto [lkey, rkey, rkey_size] = ucx_resources->context().register_memory_with_rkey(ptr, bytes);
            std::lock_guard<decltype(m_host_mutex)> l(m_host_mutex);
            m_host_regcache.add_block({ptr, bytes, lkey, rkey, rkey_size});
        }

        return ptr;
    };

    srf::memory::deallocate_callback_t host_dealloc_cb = [this, &ucx_resources](
                                                             void* ptr, std::size_t bytes, void* user_data) {
        if (ucx_resources)
        {
            std::lock_guard<decltype(m_host_mutex)> lock(m_host_mutex);
            const auto* block = m_host_regcache.find_block(ptr);
            CHECK(block);
            ucx_resources->context().unregister_memory(block->local_handle(), block->remote_handle());
            if (bytes == 0)
            {
                bytes = block->bytes();
            }
        }

        VLOG(5) << "host deallocation:" << ptr << "; size=" << bytes_to_string(bytes);

        m_host_raw->deallocate(ptr, bytes);
    };

    host_callback =
        std::make_unique<srf::memory::callback_memory_resource>(host_alloc_cb, host_dealloc_cb, m_host_raw->kind());

    if (false /* use pool */)  // NOLINT
    {
        // if we are using a pooling/arena resource, then construct it from the callback resource

        m_host_resource = srf::memory::make_unique_resource<srf::memory::arena_resource>(std::move(host_callback));

        if (device_callback)
        {
            m_device_resource =
                srf::memory::make_unique_resource<srf::memory::arena_resource>(std::move(device_callback));
        }
    }
    else
    {
        // otherwise, use the callback resource as the primary resource
        m_host_resource   = std::move(host_callback);
        m_device_resource = std::move(device_callback);
    }
}

}  // namespace srf::internal::memory
