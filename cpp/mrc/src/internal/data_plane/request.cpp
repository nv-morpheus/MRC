/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/data_plane/request.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <ucp/api/ucp.h>

#include <ostream>

namespace mrc::data_plane {

Request::Request() = default;

Request::~Request()
{
    CHECK(this->is_complete()) << "A Request that is in use is being destroyed";

    if (m_rkey != nullptr)
    {
        ucp_rkey_destroy(reinterpret_cast<ucp_rkey_h>(m_rkey));
        m_rkey = nullptr;
    }

    if (m_request != nullptr)
    {
        ucp_request_free(m_request);
        m_request = nullptr;
    }
}

void Request::reset()
{
    m_state   = State::Init;
    m_request = nullptr;
}

bool Request::is_complete()
{
    return m_state != State::Init && m_state != State::Running;
}

bool Request::await_complete()
{
    CHECK(m_state > State::Init);
    while (m_state == State::Running)
    {
        boost::this_fiber::yield();
    }

    if (m_state == State::OK)
    {
        reset();
        return true;
    }

    if (m_state == State::Cancelled)
    {
        reset();
        return false;
    }

    LOG(FATAL) << "error in ucx callback";
}

}  // namespace mrc::data_plane
