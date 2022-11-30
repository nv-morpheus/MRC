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

#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/channel/types.hpp"
#include "mrc/core/watcher.hpp"

#include <cstddef>

namespace mrc::channel {

std::size_t default_channel_size();
void set_default_channel_size(std::size_t default_size);

struct ChannelBase
{
    virtual ~ChannelBase() = 0;
};

/**
 * @brief Primary data transport layer between MRC Pipeline components responsible for handling backpressure.
 *
 * Channel is a fundamental component of the MRC library which exposes a writer/sender interface via Ingress and a
 * reader/receiver interface via Egress.
 *
 * A Channel has two primary purposes:
 * - a transport layer between components
 * - provide in-transit logic on how to handle backpressure
 *
 * When data is written to a Channel's Ingress, the channel's implementation can evaluate the state of the channel
 * with regards to backpressure and based on the provided implementation logic perform some action if backpressure
 * is detected.
 *
 * The channel interface was designed to mimic boost::fiber::buffered_channel but allow for alternative
 * implementations such as RecentChannel and NullChannel.
 *
 * @note While Channel provides an close_channel and is_channel_closed method, these are rarely used in the MRC code
 * base. A Channel is most often owned by an object which then exposed a custom Ingress to that channel for which
 * the contract guarantees the channels is open and direct closure by writers is not allowed.
 *
 * @tparam T
 */
template <typename T>
class Channel : public Ingress<T>, public Egress<T>, public Watchable, public ChannelBase
{
  public:
    ~Channel() override = default;

    inline Status await_write(T&& t) final;
    using Ingress<T>::await_write;

    inline Status await_read(T& t) final;
    Status await_read_until(T& t, const time_point_t& tp) final;
    Status try_read(T& t) final;

    void close_channel();
    bool is_channel_closed() const;

  private:
    virtual Status do_await_write(T&&) = 0;

    virtual Status do_await_read(T&)                            = 0;
    virtual Status do_await_read_until(T&, const time_point_t&) = 0;
    virtual Status do_try_read(T&)                              = 0;

    virtual void do_close_channel()           = 0;
    virtual bool do_is_channel_closed() const = 0;
};

template <typename T>
inline Status Channel<T>::await_write(T&& t)
{
    WATCHER_PROLOGUE(WatchableEvent::channel_write);
    auto rc = do_await_write(std::move(t));
    WATCHER_EPILOGUE(WatchableEvent::channel_write, rc == Status::success);
    return rc;
}

template <typename T>
inline Status Channel<T>::await_read(T& t)
{
    WATCHER_PROLOGUE(WatchableEvent::channel_read);
    auto rc = do_await_read(t);
    WATCHER_EPILOGUE(WatchableEvent::channel_read, rc == Status::success);
    return rc;
}

template <typename T>
inline Status Channel<T>::await_read_until(T& t, const time_point_t& tp)
{
    WATCHER_PROLOGUE(WatchableEvent::channel_read);
    auto rc = do_await_read_until(t, tp);
    WATCHER_EPILOGUE(WatchableEvent::channel_read, rc == Status::success);
    return rc;
}

template <typename T>
inline Status Channel<T>::try_read(T& t)
{
    WATCHER_PROLOGUE(WatchableEvent::channel_read);
    auto rc = do_try_read(t);
    WATCHER_EPILOGUE(WatchableEvent::channel_read, rc == Status::success);
    return rc;
}

template <typename T>
inline void Channel<T>::close_channel()
{
    do_close_channel();
}

template <typename T>
inline bool Channel<T>::is_channel_closed() const
{
    return do_is_channel_closed();
}

}  // namespace mrc::channel

namespace mrc {

template <typename T>
using Channel = channel::Channel<T>;  // NOLINT

}  // namespace mrc
