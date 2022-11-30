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

#include "internal/ucx/context.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucs/type/status.h>  // for ucs_status_string, UCS_OK

#include <cstring>
#include <new>        // for bad_alloc
#include <ostream>    // for logging
#include <stdexcept>  // for runtime_error
#include <tuple>      // for make_tuple, tuple

namespace mrc::internal::ucx {

Context::Context()
{
    ucp_config_t* cfg = nullptr;
    ucp_params_t ucp_params;
    std::memset(&ucp_params, 0, sizeof(ucp_params));

    auto status = ucp_config_read(nullptr, nullptr, &cfg);
    if (status != UCS_OK)
    {
        LOG(ERROR) << "ucp_config_read error: " << ucs_status_string(status);
        throw std::runtime_error("ucp_config_read failed");
    }

    // UCP initialization
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;  // | UCP_PARAM_FIELD_MT_WORKERS_SHARED;

    // add rdma and am flags here
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA;

    // MT_WORKERS_SHARED could be true if the comms and event workers are on different threads
    // ucp_params.mt_workers_shared = 1;

    status = ucp_init(&ucp_params, cfg, &m_handle);
    if (status != UCS_OK)
    {
        LOG(ERROR) << "ucp_init failed: " << ucs_status_string(status);
        throw std::runtime_error("ucp_init failed");
    }
}

Context::~Context()
{
    VLOG(5) << "destroying ucp context";
    ucp_cleanup(m_handle);
}

ucp_mem_h Context::register_memory(const void* address, std::size_t length)
{
    ucp_mem_map_params params;
    std::memset(&params, 0, sizeof(params));

    params.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;  // | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;

    CHECK(address);
    params.address = const_cast<void*>(address);
    params.length  = length;
    // params.memory_type = memory_type(ptr.type());
    // params.flags = UCP_MEM_MAP_FIXED;

    ucp_mem_h handle;

    auto status = ucp_mem_map(m_handle, &params, &handle);
    if (status != UCS_OK)
    {
        LOG(ERROR) << "ucp_mem_map failed - " << ucs_status_string(status);
        throw std::bad_alloc();
    }

    return handle;
}

std::tuple<ucp_mem_h, void*, std::size_t> Context::register_memory_with_rkey(const void* address, std::size_t length)
{
    void* rkey_buffer = nullptr;
    std::size_t buffer_size;

    auto* handle = register_memory(address, length);

    auto status = ucp_rkey_pack(m_handle, handle, &rkey_buffer, &buffer_size);
    if (status != UCS_OK)
    {
        LOG(FATAL) << "ucp_rkey_pack failed - " << ucs_status_string(status);
    }

    return std::make_tuple(handle, rkey_buffer, buffer_size);
}

void Context::unregister_memory(ucp_mem_h handle, void* rbuffer)
{
    if (rbuffer != nullptr)
    {
        ucp_rkey_buffer_release(rbuffer);
    }
    if (handle != nullptr)
    {
        auto status = ucp_mem_unmap(m_handle, handle);
        if (status != UCS_OK)
        {
            LOG(FATAL) << "ucp_mem_unmap failed - " << ucs_status_string(status);
        }
    }
}

}  // namespace mrc::internal::ucx
