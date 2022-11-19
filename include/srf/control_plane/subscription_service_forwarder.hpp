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

#include "srf/control_plane/api.hpp"

namespace srf::control_plane {

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

    void request_stop() override
    {
        service().request_stop();
    }

    bool is_live() const override
    {
        return service().is_live();
    }

    void await_join() override
    {
        service().await_join();
    }

  private:
    virtual ISubscriptionService& service() const = 0;
};

}  // namespace srf::control_plane
