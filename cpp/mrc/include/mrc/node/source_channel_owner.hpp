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

#include "mrc/edge/edge_channel.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/source_properties.hpp"

#include <memory>

namespace mrc::node {

/**
 * @brief Extends SourceProperties to hold a channel ingress which is the writing interface to an edge.
 *
 * @tparam T
 */
template <typename T>
class SourceChannelOwner : public virtual SourceProperties<T>
{
  public:
    ~SourceChannelOwner() override = default;

    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        edge::EdgeChannel<T> edge_channel(std::move(channel));

        this->do_set_channel(edge_channel);
    }

  protected:
    SourceChannelOwner() = default;

    void do_set_channel(edge::EdgeChannel<T>& edge_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_writable_edge
        auto channel_reader = edge_channel.get_reader();
        auto channel_writer = edge_channel.get_writer();

        channel_reader->add_connector([this, channel_writer]() {
            // Finally, set the other half as the connected edge to allow writers the ability to push to the channel.
            // Only do this after a full connection has been made to avoid writing to a channel that will never be
            // read from.
            SourceProperties<T>::init_connected_edge(channel_writer);
        });

        SourceProperties<T>::init_owned_edge(channel_reader);
    }
};

}  // namespace mrc::node
