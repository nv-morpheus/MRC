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

#pragma once

#include "mrc/node/forward.hpp"
#include "mrc/node/rx_subscribable.hpp"

#include <glog/logging.h>
#include <rxcpp/rx-subscription.hpp>
#include <rxcpp/rx.hpp>

#include <memory>

namespace mrc::node {

class RxExecute
{
  public:
    RxExecute() = delete;
    RxExecute(std::unique_ptr<RxSubscribable> subscribable) : m_subscribable(std::move(subscribable))
    {
        CHECK(m_subscribable);
    }

    rxcpp::subscription subscribe()
    {
        rxcpp::composite_subscription subscription;
        m_subscribable->subscribe(subscription);
        return subscription;
    }

    void subscribe(rxcpp::composite_subscription& subscription)
    {
        m_subscribable->subscribe(subscription);
    }

    template <typename T>
    [[nodiscard]] const T& subscribable_as() const
    {
        auto node = dynamic_cast<T*>(m_subscribable.get());
        CHECK(node);
        return *node;
    }

  private:
    std::unique_ptr<RxSubscribable> m_subscribable;
};

}  // namespace mrc::node
