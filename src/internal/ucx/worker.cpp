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

#include "internal/ucx/worker.hpp"

#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"

#include "srf/types.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp.h>           // for ucp_*
#include <ucp/api/ucp_def.h>       // for ucp_worker_h
#include <ucs/type/status.h>       // for ucs_status_string, UCS_OK
#include <ucs/type/thread_mode.h>  // for UCS_THREAD_MODE_MULTI

#include <cstring>  // for memset
#include <memory>
#include <ostream>    // for logging
#include <stdexcept>  // for runtime_error
#include <string>
#include <utility>

namespace srf::internal::ucx {

Worker::Worker(Handle<Context> context) : m_context(std::move(context)), m_address_pointer(nullptr), m_address_length(0)
{
    CHECK(m_context) << "null context detected when creating ucx worker";

    ucp_worker_params_t worker_params;
    std::memset(&worker_params, 0, sizeof(worker_params));

    // TODO(unknown): overtime determine the access pattern by fibers
    // _SINGLE states that only the thread that created can access which could imply thread_local storage
    // used in the implementation. Even a serialized fiber could fail in that scenario if the fiber is
    // executing on a different thread. Swifts @MainActor would be nice here.
    // We will start with _MULTI and hopefully be able to drop down to _SERIALIZED.
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    auto status = ucp_worker_create(m_context->handle(), &worker_params, &m_handle);
    if (status != UCS_OK)
    {
        LOG(ERROR) << "ucp_worker_create failed: " << ucs_status_string(status);
        throw std::runtime_error("ucp_worker_create failed");
    }
}

Worker::~Worker()
{
    release_address();

    VLOG(5) << "destroying ucp worker";
    ucp_worker_destroy(m_handle);
}

Handle<Endpoint> Worker::create_endpoint(WorkerAddress remote_address)
{
    auto casted_this = std::static_pointer_cast<Worker>(shared_from_this());
    return std::make_shared<Endpoint>(casted_this, remote_address);
}

unsigned Worker::progress()
{
    return ucp_worker_progress(m_handle);
}

const std::string& Worker::address()
{
    if (m_address_pointer == nullptr)
    {
        auto status = ucp_worker_get_address(m_handle, &m_address_pointer, &m_address_length);
        if (status != UCS_OK)
        {
            LOG(FATAL) << "ucp_worker_get_address failed - " << ucs_status_string(status);
        }

        m_address.resize(m_address_length);
        std::memcpy(m_address.data(), m_address_pointer, m_address_length);
    }
    return m_address;
}

void Worker::release_address()
{
    if (m_address_pointer != nullptr)
    {
        VLOG(5) << "releasing ucp worker address";
        ucp_worker_release_address(m_handle, m_address_pointer);
        m_address_pointer = nullptr;
        m_address.resize(0);
        m_address_length = m_address.size();
    }
}

Context& Worker::context()
{
    CHECK(m_context);
    return *m_context;
}

}  // namespace srf::internal::ucx
