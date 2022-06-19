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

#include "internal/service.hpp"

#include <glog/logging.h>

#include <ostream>

namespace srf::internal {

Service::~Service()
{
    auto state = this->state();
    CHECK(state == ServiceState::Initialized || state == ServiceState::Completed);
}

void Service::service_start()
{
    if (forward_state(ServiceState::Running))
    {
        do_service_start();
    }
}

void Service::service_await_live()
{
    do_service_await_live();
}

void Service::service_stop()
{
    if (forward_state(ServiceState::Stopping))
    {
        do_service_stop();
    }
}

void Service::service_kill()
{
    if (forward_state(ServiceState::Killing))
    {
        do_service_kill();
    }
}

void Service::service_await_join()
{
    if (forward_state(ServiceState::Awaiting))
    {
        do_service_await_join();
        forward_state(ServiceState::Completed);
    }
}

const ServiceState& Service::state() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_state;
}

bool Service::forward_state(ServiceState new_state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_state <= new_state) << "invalid ServiceState requested; ServiceState is only allowed to advance";
    if (m_state < new_state)
    {
        m_state = new_state;
        return true;
    }
    return false;
}

void Service::call_in_destructor()
{
    auto state = this->state();

    if (state <= ServiceState::Initialized)
    {
      return;
    }

    if (state == ServiceState::Running)
    {
        LOG(ERROR) << "service was not stopped/killed before being destructed; issuing kill";
        service_kill();
    }

    if (state != ServiceState::Completed)
    {
        LOG(ERROR) << "service was not joined before being destructed; issuing join";
        service_await_join();
    }
}

}  // namespace srf::internal
