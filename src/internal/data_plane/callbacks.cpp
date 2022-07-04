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

#include "internal/data_plane/callbacks.hpp"

#include "internal/data_plane/request.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp.h>

namespace srf::internal::data_plane {

void Callbacks::send(void* request, ucs_status_t status, void* user_data)
{
    DVLOG(10) << "send callback start for request " << request;

    DCHECK(user_data);
    auto* user_req = static_cast<Request*>(user_data);
    DCHECK(user_req->m_state == Request::State::Running);

    if (user_req->m_rkey != nullptr)
    {
        ucp_rkey_destroy(reinterpret_cast<ucp_rkey_h>(user_req->m_rkey));
    }

    if (status == UCS_OK)
    {
        ucp_request_free(request);
        user_req->m_request = nullptr;
        user_req->m_state   = Request::State::OK;
    }

    else if (status == UCS_ERR_CANCELED)
    {
        ucp_request_free(request);
        user_req->m_request = nullptr;
        user_req->m_state   = Request::State::Cancelled;
    }
    else
    {
        // todo(ryan) - set the promise exception ptr
        LOG(FATAL) << "data_plane: pre_posted_recv_callback failed with status: " << ucs_status_string(status);
        user_req->m_state = Request::State::Error;
    }
}

void Callbacks::recv(void* request, ucs_status_t status, const ucp_tag_recv_info_t* msg_info, void* user_data)
{
    DCHECK(user_data);
    auto* user_req = static_cast<Request*>(user_data);
    DCHECK(user_req->m_state == Request::State::Running);

    if (status == UCS_OK)  // cpp20 [[likely]]
    {
        ucp_request_free(request);
        user_req->m_request = nullptr;
        user_req->m_state   = Request::State::OK;
    }
    else if (status == UCS_ERR_CANCELED)
    {
        ucp_request_free(request);
        user_req->m_request = nullptr;
        user_req->m_state   = Request::State::Cancelled;
    }
    else
    {
        // todo(ryan) - set the promise exception ptr
        LOG(FATAL) << "data_plane: pre_posted_recv_callback failed with status: " << ucs_status_string(status);
        user_req->m_state = Request::State::Error;
    }
}

}  // namespace srf::internal::data_plane
