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

#include "internal/data_plane/request.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>

namespace srf::internal::data_plane {

Request::Request() = default;

Request::~Request()
{
    CHECK(m_state == State::Init) << "A Request that is in use is being destroyed";
}

void Request::reset()
{
    m_state   = State::Init;
    m_request = nullptr;
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

}  // namespace srf::internal::data_plane
