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

#include "srf/channel/ingress.hpp"
#include "srf/node/edge_properties.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_channel_base.hpp"

namespace srf::node {

template <typename T>
class Queue final : public SinkChannelBase<T>, public SinkProperties<T>, public ChannelProvider<T>
{
  public:
    Queue()        = default;
    ~Queue() final = default;

  private:
    // SinkProperties<T> - aka IngressProvider
    std::shared_ptr<channel::Ingress<T>> channel_ingress() final
    {
        return SinkChannelBase<T>::ingress_channel();
    }

    // ChannelProvider
    std::shared_ptr<channel::Channel<T>> channel() final
    {
        return SinkChannelBase<T>::channel();
    }
};

}  // namespace srf::node
