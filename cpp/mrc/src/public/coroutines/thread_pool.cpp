/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/thread_pool.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace mrc::coroutines {

thread_local ThreadPool* ThreadPool::m_self{nullptr};
thread_local std::size_t ThreadPool::m_thread_id{0};

ThreadPool::Operation::Operation(ThreadPool& tp) noexcept : m_thread_pool(tp) {}

auto ThreadPool::Operation::await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> void
{
// create span to measure the time spent in the scheduler
#if srf_ENABLE_OTEL_TRACE
    m_span = mrc::trace::get_tracer()->StartSpan("schedule to thread_pool");
    if (m_span->Isrfcording())
    {
        m_span->AddEvent("suspend coroutine for scheduling on " + m_thread_pool.description(),
                         {{"thread.id", mrc::this_thread::get_id()}});
        m_span->SetAttribute("component", "mrc::thread_pool");
    }
#endif
    // DVLOG(10) << "suspend scheduling operation on " << mrc::this_thread::get_id();

    // suspend thread local state
    // ThreadLocalContext::suspend_thread_local_context();

    // capture the coroutine handle and schedule it to be resumed
    m_awaiting_coroutine = awaiting_coroutine;
    m_thread_pool.schedule_impl(m_awaiting_coroutine);
}

auto ThreadPool::Operation::await_resume() noexcept -> void
{
    // restore thread local state
    // ThreadLocalContext::resume_thread_local_context();

    // complete the span recording the time spent scheduling
#if srf_ENABLE_OTEL_TRACE
    if (m_span->Isrfcording())
    {
        m_span->AddEvent("resuming coroutine scheduled on " + m_thread_pool.description(),
                         {{"thread.id", mrc::this_thread::get_id()}});
    }
    m_span->End();
#endif
    // DVLOG(10) << "resuming schedule operation on " << mrc::this_thread::get_id();
}

ThreadPool::ThreadPool(Options opts) : m_opts(std::move(opts))
{
    if (m_opts.description.empty())
    {
        std::stringstream ss;
        ss << "thread_pool_" << this;
        m_opts.description = ss.str();
    }

    m_threads.reserve(m_opts.thread_count);

    for (uint32_t i = 0; i < m_opts.thread_count; ++i)
    {
        m_threads.emplace_back([this, i](std::stop_token st) {
            executor(std::move(st), i);
        });
    }
}

ThreadPool::~ThreadPool()
{
    shutdown();
}

auto ThreadPool::schedule() -> Operation
{
    if (!m_shutdown_requested.load(std::memory_order::relaxed))
    {
        m_size.fetch_add(1, std::memory_order::release);
        return Operation{*this};
    }

    throw std::runtime_error("coroutines::ThreadPool is shutting down, unable to schedule new tasks.");
}

auto ThreadPool::resume(std::coroutine_handle<> handle) noexcept -> void
{
    if (handle == nullptr)
    {
        return;
    }

    m_size.fetch_add(1, std::memory_order::release);
    schedule_impl(handle);
}

auto ThreadPool::shutdown() noexcept -> void
{
    // Only allow shutdown to occur once.
    if (!m_shutdown_requested.exchange(true, std::memory_order::acq_rel))
    {
        for (auto& thread : m_threads)
        {
            thread.request_stop();
        }

        for (auto& thread : m_threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
    }
}

auto ThreadPool::executor(std::stop_token stop_token, std::size_t idx) -> void
{
    m_self      = this;
    m_thread_id = idx;

    if (m_opts.on_thread_start_functor != nullptr)
    {
        m_opts.on_thread_start_functor(idx);
    }

    while (!stop_token.stop_requested())
    {
        // Wait until the queue has operations to execute or shutdown has been requested.
        while (true)
        {
            std::unique_lock<std::mutex> lk{m_wait_mutex};
            m_wait_cv.wait(lk, stop_token, [this] {
                return !m_queue.empty();
            });
            if (m_queue.empty())
            {
                lk.unlock();  // would happen on scope destruction, but being explicit/faster(?)
                break;
            }

            auto handle = m_queue.front();
            m_queue.pop_front();

            lk.unlock();  // Not needed for processing the coroutine.

            handle.resume();
            m_size.fetch_sub(1, std::memory_order::release);
        }
    }

    if (m_opts.on_thread_stop_functor != nullptr)
    {
        m_opts.on_thread_stop_functor(idx);
    }
}

auto ThreadPool::schedule_impl(std::coroutine_handle<> handle) noexcept -> void
{
    if (handle == nullptr)
    {
        return;
    }

    {
        std::scoped_lock lk{m_wait_mutex};
        m_queue.emplace_back(handle);
    }

    m_wait_cv.notify_one();
}

auto ThreadPool::from_current_thread() -> ThreadPool*
{
    return m_self;
}

auto ThreadPool::get_thread_id() -> std::size_t
{
    return m_thread_id;
}

const std::string& ThreadPool::description() const
{
    return m_opts.description;
}

}  // namespace mrc::coroutines
