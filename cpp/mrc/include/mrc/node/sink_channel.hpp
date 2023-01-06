/**
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

#pragma once

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/constants.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge.hpp"
#include "mrc/node/edge_channel.hpp"
#include "mrc/node/edge_properties.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_channel_base.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <mutex>

namespace mrc::node {

/**
 * @brief Extends SinkProperties to hold a Channel and provide SinkProperties access the the channel ingress
 *
 * @tparam T
 */
template <typename T>
class SinkChannel : public virtual SinkProperties<T>
{
  public:
    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        EdgeChannel<T> edge_channel(std::move(channel));

        this->do_set_channel(edge_channel);
    }

  protected:
    SinkChannel() = default;

    void do_set_channel(EdgeChannel<T>& edge_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_readable+edge
        auto channel_reader = edge_channel.get_reader();
        auto channel_writer = edge_channel.get_writer();

        SinkProperties<T>::init_owned_edge(channel_writer);

        // Finally, set the other half to m_set_edge to allow using the channel without it being deleted. If set_edge()
        // is called, then this will be overwritten
        SinkProperties<T>::init_connected_edge(channel_reader);
    }
};

// template <typename T>
// class SinkChannelReadable : public SinkChannel<T>
// {
//   public:
//     using SinkChannel<T>::egress;
// };

}  // namespace mrc::node
