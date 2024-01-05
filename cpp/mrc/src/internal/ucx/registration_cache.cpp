/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/ucx/registration_cache.hpp"

#include <ucxx/api.h>

namespace mrc::ucx {

RegistrationCache2::RegistrationCache2(std::shared_ptr<ucxx::Context> context) : m_context(std::move(context))
{
    CHECK(m_context);
}

void RegistrationCache2::add_block(const void* addr, std::size_t bytes)
{
    DCHECK(addr && bytes);
    auto [lkey, rkey, rkey_size] = this->register_memory_with_rkey(addr, bytes);
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_blocks.add_block({addr, bytes, lkey, rkey, rkey_size});
}

std::size_t RegistrationCache2::drop_block(const void* addr, std::size_t bytes)
{
    const auto* block = m_blocks.find_block(addr);
    CHECK(block);
    bytes = block->bytes();
    this->unregister_memory(block->local_handle(), block->remote_handle());
    m_blocks.drop_block(addr);
    return bytes;
}

std::optional<ucx::MemoryBlock> RegistrationCache2::lookup(const void* addr) const noexcept
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    const auto* ptr = m_blocks.find_block(addr);
    if (ptr == nullptr)
    {
        return std::nullopt;
    }
    return {*ptr};
}

std::optional<ucx::MemoryBlock> RegistrationCache2::lookup(uintptr_t addr) const noexcept
{
    return this->lookup(reinterpret_cast<const void*>(addr));
}

ucp_mem_h RegistrationCache2::register_memory(const void* address, std::size_t bytes)
{
    ucp_mem_map_params params;
    std::memset(&params, 0, sizeof(params));

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;  // | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;

    CHECK(address);
    params.address = const_cast<void*>(address);
    params.length  = bytes;
    // params.memory_type = memory_type(ptr.type());
    // params.flags = UCP_MEM_MAP_FIXED;

    ucp_mem_h handle;

    auto status = ucp_mem_map(m_context->getHandle(), &params, &handle);
    if (status != UCS_OK)
    {
        LOG(ERROR) << "ucp_mem_map failed - " << ucs_status_string(status);
        throw std::bad_alloc();
    }

    return handle;
}

std::tuple<ucp_mem_h, void*, std::size_t> RegistrationCache2::register_memory_with_rkey(const void* address,
                                                                                        std::size_t bytes)
{
    void* rkey_buffer = nullptr;
    std::size_t buffer_size;

    auto* handle = this->register_memory(address, bytes);

    auto status = ucp_rkey_pack(m_context->getHandle(), handle, &rkey_buffer, &buffer_size);
    if (status != UCS_OK)
    {
        LOG(FATAL) << "ucp_rkey_pack failed - " << ucs_status_string(status);
    }

    return std::make_tuple(handle, rkey_buffer, buffer_size);
}

void RegistrationCache2::unregister_memory(ucp_mem_h handle, void* rbuffer)
{
    if (rbuffer != nullptr)
    {
        ucp_rkey_buffer_release(rbuffer);
    }
    if (handle != nullptr)
    {
        auto status = ucp_mem_unmap(m_context->getHandle(), handle);
        if (status != UCS_OK)
        {
            LOG(FATAL) << "ucp_mem_unmap failed - " << ucs_status_string(status);
        }
    }
}
}  // namespace mrc::ucx