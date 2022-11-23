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

#include "internal/ucx/endpoint.hpp"

#include "internal/ucx/common.hpp"
#include "internal/ucx/remote_registration_cache.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/types.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp.h>      // for ucp_ep_close_nb, ucp_ep_create, UCP_EP_...
#include <ucp/api/ucp_def.h>  // for ucp_ep_params_t, ucp_address_t, ucp_ep
#include <ucs/type/status.h>  // for ucs_status_string, UCS_OK, UCS_PTR_STATUS

#include <cstring>  // for memset
#include <memory>   // Handle is a shared_ptr
#include <ostream>  // for logging
#include <utility>

namespace mrc::internal::ucx {

Endpoint::Endpoint(Handle<Worker> local_worker, WorkerAddress remote_worker) : m_worker(std::move(local_worker))
{
    ucp_ep_params_t params;
    std::memset(&params, 0, sizeof(params));

    params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    params.address    = reinterpret_cast<const ucp_address_t*>(remote_worker.data());

    auto status = ucp_ep_create(m_worker->handle(), &params, &m_handle);
    if (status != UCS_OK)
    {
        LOG(FATAL) << "ucp_ep_create failed: " << ucs_status_string(status);
    }

    m_registration_cache = std::make_unique<RemoteRegistrationCache>(m_handle);
}

Endpoint::~Endpoint()
{
    DVLOG(10) << "closing ep";
    auto* request = ucp_ep_close_nb(m_handle, UCP_EP_CLOSE_MODE_FLUSH);

    // According to the docs a null response is considered OK
    if (request == nullptr)
    {
        return;
    }
    if (UCS_PTR_IS_ERR(request))
    {
        auto status = UCS_PTR_STATUS(request);
        LOG(WARNING) << "failed to close ep: " << ucs_status_string(status);
        return;
    }

    ucs_status_t status;
    do
    {
        m_worker->progress();
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);

    ucp_request_free(request);
}

RemoteRegistrationCache& Endpoint::registration_cache()
{
    CHECK(m_registration_cache);
    return *m_registration_cache;
}

const RemoteRegistrationCache& Endpoint::registration_cache() const
{
    CHECK(m_registration_cache);
    return *m_registration_cache;
}

}  // namespace mrc::internal::ucx
