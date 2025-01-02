/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/node/source_properties.hpp"

#include <type_traits>

namespace mrc::node {

/**
 * @brief This object is a utility node that can be used as an entrypoint into a chain of nodes. It functions very
 * similarly to an RxSource except it does not have a progress engine. This node is driven by external calls to
 * `await_write`. This node can be used to keep downstream connections from closing (i.e. persist downstream nodes even
 * after all upstream connections have completed), can be used to insert items into a runnable stream on-demand, and is
 * very useful for testing.
 *
 * Similar to every other node, this must be connected with `make_edge` and will hold the downstream nodes open as long
 * as this object is alive.
 *
 * @tparam T
 */
template <typename T>
class WritableEntrypoint<T, std::enable_if_t<std::is_const_v<T>>> : public WritableAcceptor<T>
{
  public:
    channel::Status await_write(T&& data)
    {
        return this->get_writable_edge()->await_write(std::move(data));
    }

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline channel::Status await_write(T data)
    {
        return await_write(std::move(data));
    }
};

template <typename T>
class WritableEntrypoint<T, std::enable_if_t<!std::is_const_v<T>>>
  : public WritableAcceptor<T>, public ReadableProvider<T>, public SourceChannelOwner<T>
{
  public:
    WritableEntrypoint()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    channel::Status await_write(T&& data)
    {
        return this->get_writable_edge()->await_write(std::move(data));
    }

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline channel::Status await_write(T data)
    {
        return await_write(std::move(data));
    }
};

}  // namespace mrc::node
