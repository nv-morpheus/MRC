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
#include "mrc/types.hpp"  // for CondV & Mutex

#include <boost/fiber/mutex.hpp>

#include <cstddef>  // for size_t
#include <deque>
#include <mutex>
#include <thread>  // for lock_guard & unique_lock

namespace mrc::channel {

template <typename T>
class RecentChannel : public Channel<T>
{
  public:
    RecentChannel(std::size_t count = default_channel_size()) : m_max_size(count) {}
    ~RecentChannel() override = default;

  private:
    Status do_await_write(T&& data) override
    {
        std::lock_guard<Mutex> lock(m_mutex);
        if (m_is_shutdown)
        {
            return Status::closed;
        }
        if (m_deque.size() >= m_max_size)
        {
            m_deque.pop_front();
        }
        m_deque.push_back(std::move(data));
        m_cv.notify_one();
        return Status::success;
    }

    Status do_await_read(T& data) override
    {
        std::unique_lock<Mutex> lock(m_mutex);
        while (m_deque.empty() && !m_is_shutdown)
        {
            m_cv.wait(lock, [this] { return m_is_shutdown || !m_deque.empty(); });
        }
        if (m_is_shutdown)
        {
            return Status::closed;
        }
        data = std::move(m_deque.front());
        m_deque.pop_front();
        return Status::success;
    }

    Status do_try_read(T& data) override
    {
        std::unique_lock<Mutex> lock(m_mutex);
        if (m_is_shutdown)
        {
            return Status::closed;
        }
        if (m_deque.empty())
        {
            return Status::empty;
        }
        data = std::move(m_deque.front());
        m_deque.pop_front();
        return Status::success;
    }

    Status do_await_read_until(T& data, const time_point_t& deadline) override
    {
        std::unique_lock<Mutex> lock(m_mutex);

        while (m_deque.empty() && !m_is_shutdown && deadline < clock_t::now())
        {
            m_cv.wait_until(lock, deadline, [this] { return m_is_shutdown || !m_deque.empty(); });
        }
        if (m_is_shutdown)
        {
            return Status::closed;
        }
        if (m_deque.empty())
        {
            return Status::timeout;
        }
        data = std::move(m_deque.front());
        m_deque.pop_front();
        return Status::success;
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
    bool m_is_shutdown{false};
    std::size_t m_max_size;
    std::deque<T> m_deque;
};

}  // namespace mrc::channel

namespace mrc {

template <typename T>
using RecentChannel = channel::RecentChannel<T>;  // NOLINT

}
