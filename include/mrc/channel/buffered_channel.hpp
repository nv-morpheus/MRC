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

#include "mrc/channel/channel.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>

namespace mrc::channel {

template <typename T>
class BufferedChannel final : public Channel<T>
{
    using status_t = boost::fibers::channel_op_status;

  public:
    BufferedChannel(std::size_t buffer_size = default_channel_size()) : m_channel(buffer_size) {}
    ~BufferedChannel() final = default;

  private:
    inline Status do_await_write(T&& val) final
    {
        return status(m_channel.push(std::move(val)));
    }

    inline Status do_await_read(T& val) final
    {
        return status(m_channel.pop(std::ref(val)));
    }

    Status do_try_read(T& val) final
    {
        return status(m_channel.try_pop(std::ref(val)));
    }

    Status do_await_read_until(T& val, const time_point_t& deadline) final
    {
        return status(m_channel.pop_wait_until(std::ref(val), deadline));
    }

    void do_close_channel() final
    {
        m_channel.close();
    }

    bool do_is_channel_closed() const final
    {
        return m_channel.is_closed();
    }

    Status status(const status_t rc)
    {
        switch (rc)
        {
        case status_t::success:
            return Status::success;
        case status_t::closed:
            return Status::closed;
        case status_t::empty:
            return Status::empty;
        case status_t::full:
            return Status::full;
        case status_t::timeout:
            return Status::timeout;
        }
        return Status::error;
    }

    boost::fibers::buffered_channel<T> m_channel;
};

}  // namespace mrc::channel

namespace mrc {

template <typename T>
using BufferedChannel = channel::BufferedChannel<T>;  // NOLINT

}
