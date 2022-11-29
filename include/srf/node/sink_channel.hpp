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

#include "srf/channel/buffered_channel.hpp"
#include "srf/channel/egress.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/constants.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_properties.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_channel_base.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/utils/type_utils.hpp"

#include <mutex>

namespace srf::node {

/**
 * @brief Extends SinkProperties to hold a Channel and provide SinkProperties access the the channel ingress
 *
 * @tparam T
 */
template <typename T>
class SinkChannel : public SinkChannelBase<T>, public SinkProperties<T>, public ChannelAcceptor<T>
{
  protected:
    SinkChannel() = default;

  private:
    using SinkChannelBase<T>::channel;
    using SinkChannelBase<T>::ingress_channel;
    using SinkChannelBase<T>::set_shared_channel;

    // implement virtual method from SinkProperties<T>
    [[nodiscard]] std::shared_ptr<channel::Ingress<T>> channel_ingress() final;

    // implement virtual method from ChannelAcceptor<T>
    void set_channel(std::shared_ptr<channel::Channel<T>> channel) final;
};

template <typename T>
std::shared_ptr<channel::Ingress<T>> SinkChannel<T>::channel_ingress()
{
    return SinkChannelBase<T>::ingress_channel();
}

template <typename T>
void SinkChannel<T>::set_channel(std::shared_ptr<Channel<T>> channel)
{
    SinkChannelBase<T>::set_shared_channel(std::move(channel));
}

template <typename T>
class SinkChannelReadable : public SinkChannel<T>
{
  public:
    using SinkChannel<T>::egress;
};

}  // namespace srf::node
