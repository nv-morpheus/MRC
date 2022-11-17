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

#include "srf/pubsub/api.hpp"

namespace srf::pubsub {

class SubscriptionService : public ISubscriptionService
{
  public:
    ~SubscriptionService() override = default;

    const std::string& service_name() const final
    {
        return service().service_name();
    }

    const std::uint64_t& tag() const final
    {
        return service().tag();
    }

    void stop() final
    {
        service().stop();
    }

    void kill() final
    {
        service().kill();
    }

    bool is_live() const final
    {
        return service().is_live();
    }

    void await_join() final
    {
        service().await_join();
    }

  private:
    virtual ISubscriptionService& service() const = 0;
};

}  // namespace srf::pubsub
