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

#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/constants.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/edge_channel.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>

namespace mrc::node {

/**
 * @brief Extends SourceProperties to hold a channel ingress which is the writing interface to an edge.
 *
 * @tparam T
 */
template <typename T>
class SourceChannel : public virtual SourceProperties<T>
{
  public:
    ~SourceChannel() override = default;

    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        EdgeChannel<T> edge_channel(std::move(channel));

        this->do_set_channel(edge_channel);
    }

  protected:
    SourceChannel() = default;

    void do_set_channel(EdgeChannel<T>& edge_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_writable_edge
        auto channel_reader = edge_channel.get_reader();
        auto channel_writer = edge_channel.get_writer();

        SourceProperties<T>::init_owned_edge(channel_reader);

        // Finally, set the other half to the connected edge to allow using the channel without it being deleted. If
        // make_edge_connection() is called, then this will be overwritten
        SourceProperties<T>::init_connected_edge(channel_writer);
    }
};

}  // namespace mrc::node
