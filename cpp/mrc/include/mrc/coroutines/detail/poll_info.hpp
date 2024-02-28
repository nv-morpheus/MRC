/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Original Source: https://github.com/jbaldwin/libcoro
 * Original License: Apache License, Version 2.0; included below
 */

/**
 * Copyright 2021 Josh Baldwin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mrc/coroutines/fd.hpp"
#include "mrc/coroutines/poll.hpp"
#include "mrc/coroutines/time.hpp"

#include <atomic>
#include <coroutine>
#include <map>
#include <optional>

namespace mrc::coroutines::detail {
/**
 * Poll Info encapsulates everything about a poll operation for the event as well as its paired
 * timeout.  This is important since coroutines that are waiting on an event or timeout do not
 * immediately execute, they are re-scheduled onto the thread pool, so its possible its pair
 * event or timeout also triggers while the coroutine is still waiting to resume.  This means that
 * the first one to happen, the event itself or its timeout, needs to disable the other pair item
 * prior to resuming the coroutine.
 *
 * Finally, its also important to note that the event and its paired timeout could happen during
 * the same epoll_wait and possibly trigger the coroutine to start twice.  Only one can win, so the
 * first one processed sets m_processed to true and any subsequent events in the same epoll batch
 * are effectively discarded.
 */
struct PollInfo
{
    using timed_events_t = std::multimap<mrc::coroutines::time_point_t, detail::PollInfo*>;

    PollInfo()  = default;
    ~PollInfo() = default;

    PollInfo(const PollInfo&)                    = delete;
    PollInfo(PollInfo&&)                         = delete;
    auto operator=(const PollInfo&) -> PollInfo& = delete;
    auto operator=(PollInfo&&) -> PollInfo&      = delete;

    struct PollAwaiter
    {
        explicit PollAwaiter(PollInfo& pi) noexcept : m_pi(pi) {}

        static auto await_ready() noexcept -> bool
        {
            return false;
        }
        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> void
        {
            m_pi.m_awaiting_coroutine = awaiting_coroutine;
            std::atomic_thread_fence(std::memory_order::release);
        }
        auto await_resume() const noexcept -> mrc::coroutines::PollStatus
        {
            return m_pi.m_poll_status;
        }

        PollInfo& m_pi;
    };

    auto operator co_await() noexcept -> PollAwaiter
    {
        return PollAwaiter{*this};
    }

    /// The file descriptor being polled on.  This is needed so that if the timeout occurs first then
    /// the event loop can immediately disable the event within epoll.
    fd_t m_fd{-1};
    /// The timeout's position in the timeout map.  A poll() with no timeout or yield() this is empty.
    /// This is needed so that if the event occurs first then the event loop can immediately disable
    /// the timeout within epoll.
    std::optional<timed_events_t::iterator> m_timer_pos{std::nullopt};
    /// The awaiting coroutine for this poll info to resume upon event or timeout.
    std::coroutine_handle<> m_awaiting_coroutine;
    /// The status of the poll operation.
    mrc::coroutines::PollStatus m_poll_status{mrc::coroutines::PollStatus::error};
    /// Did the timeout and event trigger at the same time on the same epoll_wait call?
    /// Once this is set to true all future events on this poll info are null and void.
    bool m_processed{false};
};

}  // namespace mrc::coroutines::detail
