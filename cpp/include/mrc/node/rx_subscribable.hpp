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

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <memory>

namespace mrc::node {

class RxSubscribable
{
  public:
    RxSubscribable()          = default;
    virtual ~RxSubscribable() = default;

  protected:
    void subscribe(rxcpp::composite_subscription& subscription)
    {
        do_subscribe(subscription);
    }

  private:
    virtual void do_subscribe(rxcpp::composite_subscription&) = 0;

    friend RxExecute;
};

}  // namespace mrc::node
