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
#include "mrc/types.hpp"  // for Mutex & CondV

#include <memory>  // for lock_guard

namespace mrc::channel {

template <typename T>
class NullChannel : public Channel<T>
{
  public:
    ~NullChannel() override = default;

  private:
    Status do_await_write(T&& t) override
    {
        if (m_is_shutdown)
        {
            return Status::closed;
        }
        return Status::success;
    }

    Status do_await_read(T& t) override
    {
        std::unique_lock<Mutex> lock(m_mutex);
        m_cv.wait(lock, [this] { return m_is_shutdown; });
        return Status::closed;
    }

    Status do_try_read(T& t) override
    {
        return Status::empty;
    }

    Status do_await_read_until(T& t, const time_point_t& deadline) override
    {
        std::unique_lock<Mutex> lock(m_mutex);
        m_cv.wait_until(lock, deadline, [this] { return m_is_shutdown; });
        return (m_is_shutdown ? Status::closed : Status::timeout);
    }

    void do_close_channel() override
    {
        std::lock_guard<Mutex> lock(m_mutex);
        m_is_shutdown = true;
        m_cv.notify_all();
    }

    bool do_is_channel_closed() const override
    {
        std::lock_guard<Mutex> lock(m_mutex);
        return m_is_shutdown;
    }

    mutable Mutex m_mutex;
    CondV m_cv;
    bool m_is_shutdown;
};

}  // namespace mrc::channel

namespace mrc {

template <typename T>
using NullChannel = channel::NullChannel<T>;  // NOLINT

}
