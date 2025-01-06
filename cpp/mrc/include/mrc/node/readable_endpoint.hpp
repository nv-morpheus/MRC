/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"

namespace mrc::node {

/**
 * @brief This object is a utility node that can be used as an endpoint of a chain of nodes. It functions very
 * similarly to an RxSink except it does not have a progress engine. This node is driven by external calls to
 * `await_read`. This node can be used to keep upstream connections from closing (i.e. persist upstream nodes even
 * after all downstream connections have completed), can be used to retrieve items from a runnable stream on-demand, and
 * is very useful for testing.
 *
 * Similar to every other node, this must be connected with `make_edge` and will hold the upstream nodes open as long
 * as this object is alive.
 *
 * @tparam T
 */
template <typename T>
class ReadableEndpoint<T, std::enable_if_t<std::is_const_v<T>>> : public ReadableAcceptor<T>
{
  public:
    channel::Status await_read(T& data)
    {
        return this->get_readable_edge()->await_read(data);
    }
};

template <typename T>
class ReadableEndpoint<T, std::enable_if_t<!std::is_const_v<T>>>
  : public ReadableAcceptor<T>, public WritableProvider<T>, public SinkChannelOwner<T>
{
  public:
    ReadableEndpoint()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    channel::Status await_read(T& data)
    {
        return this->get_readable_edge()->await_read(data);
    }
};

}  // namespace mrc::node
