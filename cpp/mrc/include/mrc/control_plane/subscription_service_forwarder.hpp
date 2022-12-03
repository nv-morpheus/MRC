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

#include "mrc/control_plane/api.hpp"

namespace mrc::control_plane {

/**
 * @brief Convenience class to forward the ISubscriptionService to an instance of an ISubscriptionService
 */
class SubscriptionServiceForwarder : public ISubscriptionService
{
  public:
    ~SubscriptionServiceForwarder() override = default;

    const std::string& service_name() const final
    {
        return service().service_name();
    }

    const std::uint64_t& tag() const final
    {
        return service().tag();
    }

    const std::string& role() const final
    {
        return service().role();
    }

    const std::set<std::string>& subscribe_to_roles() const final
    {
        return service().subscribe_to_roles();
    }

    void await_start() override
    {
        service().await_start();
    }

    void request_stop() override
    {
        service().request_stop();
    }

    void await_join() override
    {
        service().await_join();
    }

    bool is_startable() const override
    {
        return service().is_startable();
    }

    bool is_live() const override
    {
        return service().is_live();
    }

  private:
    virtual ISubscriptionService& service() const = 0;
};

}  // namespace mrc::control_plane
