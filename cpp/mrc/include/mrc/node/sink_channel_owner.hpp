/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/edge/edge_channel.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"

#include <mutex>

namespace mrc::node {

/**
 * @brief Extends SinkProperties to hold a Channel and provide SinkProperties access the the channel ingress
 *
 * @tparam T
 */
template <typename T>
class SinkChannelOwner : public virtual SinkProperties<T>
{
  public:
    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        edge::EdgeChannel<T> edge_channel(std::move(channel));

        this->do_set_channel(std::move(edge_channel));
    }

  protected:
    SinkChannelOwner() = default;

    void do_set_channel(edge::EdgeChannel<T> edge_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_readable+edge
        auto channel_reader = edge_channel.get_reader();
        auto channel_writer = edge_channel.get_writer();

        channel_writer->add_connector([this, channel_reader]() {
            // Finally, set the other half as the connected edge to allow readers the ability to pull from the channel.
            // Only do this after a full connection has been made to avoid reading from a channel that will never be
            // written to
            SinkProperties<T>::init_connected_edge(channel_reader);
        });

        SinkProperties<T>::init_owned_edge(channel_writer);
    }
};

}  // namespace mrc::node
