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

#pragma once

#include "mrc/core/expected.hpp"
#include "mrc/coroutines/schedule_policy.hpp"
#include "mrc/coroutines/thread_local_context.hpp"
#include "mrc/coroutines/thread_pool.hpp"

#include <glog/logging.h>

#include <atomic>
#include <coroutine>
#include <mutex>
#include <optional>
#include <vector>

namespace mrc::coroutines {

enum class RingBufferOpStatus
{
    Success,
    Stopped,
};

class Semaphore
{
    using mutex_type = std::mutex;

  public:
    struct Options
    {
        // capacity of ring buffer
        std::size_t capacity{8};

        // when there is an awaiting reader, the active execution context of the next writer will resume the awaiting
        // reader, the schedule_policy_t dictates how that is accomplished.
        SchedulePolicy reader_policy{SchedulePolicy::Reschedule};

        // when there is an awaiting writer, the active execution context of the next reader will resume the awaiting
        // writer, the producder_policy_t dictates how that is accomplished.
        SchedulePolicy writer_policy{SchedulePolicy::Reschedule};
    };

    /**
     * @throws std::runtime_error If `num_elements` == 0.
     */
    explicit Semaphore(Options opts = {}) :
      m_num_elements(opts.capacity),
      m_writer_policy(opts.writer_policy),
      m_reader_policy(opts.reader_policy)
    {
        if (m_num_elements == 0)
        {
            throw std::runtime_error{"num_elements cannot be zero"};
        }
    }

    ~Semaphore()
    {
        // // Wake up anyone still using the ring buffer.
        // notify_waiters();
    }

    Semaphore(const Semaphore&) = delete;
    Semaphore(Semaphore&&)      = delete;

    auto operator=(const Semaphore&) noexcept -> Semaphore& = delete;
    auto operator=(Semaphore&&) noexcept -> Semaphore&      = delete;

    struct WriteOperation : ThreadLocalContext
    {
        WriteOperation(Semaphore& rb) : m_rb(rb), m_policy(m_rb.m_writer_policy) {}

        auto await_ready() noexcept -> bool
        {
            // return immediate if the buffer is closed
            if (m_rb.m_stopped.load(std::memory_order::acquire))
            {
                m_stopped = true;
                return true;
            }

            // start a span to time the write - this would include time suspended if the buffer is full
            // m_write_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});

            // the lock is owned by the operation, not scoped to the await_ready function
            m_lock = std::unique_lock(m_rb.m_mutex);
            return m_rb.try_write_locked(m_lock);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            auto lock = std::move(m_lock);  // use raii

            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine = awaiting_coroutine;
            m_next               = m_rb.m_write_waiters;
            m_rb.m_write_waiters = this;
            return true;
        }

        /**
         * @return write_result
         */
        auto await_resume() -> RingBufferOpStatus
        {
            ThreadLocalContext::resume_thread_local_context();
            return (!m_stopped ? RingBufferOpStatus::Success : RingBufferOpStatus::Stopped);
        }

        WriteOperation& use_scheduling_policy(SchedulePolicy policy) &
        {
            m_policy = policy;
            return *this;
        }

        WriteOperation use_scheduling_policy(SchedulePolicy policy) &&
        {
            m_policy = policy;
            return std::move(*this);
        }

        WriteOperation& resume_immediately() &
        {
            m_policy = SchedulePolicy::Immediate;
            return *this;
        }

        WriteOperation resume_immediately() &&
        {
            m_policy = SchedulePolicy::Immediate;
            return std::move(*this);
        }

        WriteOperation& resume_on(ThreadPool* thread_pool) &
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return *this;
        }

        WriteOperation resume_on(ThreadPool* thread_pool) &&
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return std::move(*this);
        }

      private:
        friend Semaphore;

        void resume()
        {
            if (m_policy == SchedulePolicy::Immediate)
            {
                set_resume_on_thread_pool(nullptr);
            }
            resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        std::unique_lock<mutex_type> m_lock;
        /// The ring buffer the element is being written into.
        Semaphore& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be written.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of write operations that are awaiting to write their element.
        WriteOperation* m_next{nullptr};
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the Semaphore, but can be overrided owner of the Awaiter
        SchedulePolicy m_policy;
        /// Span to measure the duration the writer spent writting data
        // trace::Handle<trace::Span> m_write_span{nullptr};
    };

    struct ReadOperation : ThreadLocalContext
    {
        explicit ReadOperation(Semaphore& rb) : m_rb(rb), m_policy(m_rb.m_reader_policy) {}

        auto await_ready() noexcept -> bool
        {
            // the lock is owned by the operation, not scoped to the await_ready function
            m_lock = std::unique_lock(m_rb.m_mutex);
            // m_read_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});
            return m_rb.try_read_locked(m_lock, this);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            auto lock = std::move(m_lock);

            // // the buffer is empty; don't suspend if the stop signal has been set.
            // if (m_rb.m_stopped.load(std::memory_order::acquire))
            // {
            //     m_stopped = true;
            //     return false;
            // }

            // m_read_span->AddEvent("buffer_empty");
            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine = awaiting_coroutine;
            m_next               = m_rb.m_read_waiters;
            m_rb.m_read_waiters  = this;
            return true;
        }

        /**
         * @return The consumed element or std::nullopt if the read has failed.
         */
        auto await_resume() -> mrc::expected<void, RingBufferOpStatus>
        {
            ThreadLocalContext::resume_thread_local_context();

            if (m_stopped)
            {
                return mrc::unexpected<RingBufferOpStatus>(RingBufferOpStatus::Stopped);
            }

            return std::move(m_e);
        }

        ReadOperation& use_scheduling_policy(SchedulePolicy policy)
        {
            m_policy = policy;
            return *this;
        }

        ReadOperation& resume_immediately()
        {
            m_policy = SchedulePolicy::Immediate;
            return *this;
        }

        ReadOperation& resume_on(ThreadPool* thread_pool)
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return *this;
        }

      private:
        friend Semaphore;

        void resume()
        {
            if (m_policy == SchedulePolicy::Immediate)
            {
                set_resume_on_thread_pool(nullptr);
            }
            resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        std::unique_lock<mutex_type> m_lock;
        /// The ring buffer to read an element from.
        Semaphore& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be consumed.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of read operations that are awaiting to read an element.
        ReadOperation* m_next{nullptr};
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the Semaphore, but can be overrided owner of the Awaiter
        SchedulePolicy m_policy;
        /// Span measure time awaiting on reading data
        // trace::Handle<trace::Span> m_read_span;
    };

    /**
     * Produces the given element into the ring buffer.  This operation will suspend until a slot
     * in the ring buffer becomes available.
     * @param e The element to write.
     */
    [[nodiscard]] auto write() -> WriteOperation
    {
        return WriteOperation{*this};
    }

    /**
     * Consumes an element from the ring buffer.  This operation will suspend until an element in
     * the ring buffer becomes available.
     */
    [[nodiscard]] auto read() -> ReadOperation
    {
        return ReadOperation{*this};
    }

    /**
     * @return The current number of elements contained in the ring buffer.
     */
    auto size() const -> size_t
    {
        std::atomic_thread_fence(std::memory_order::acquire);
        return m_used;
    }

    /**
     * @return True if the ring buffer contains zero elements.
     */
    auto empty() const -> bool
    {
        return size() == 0;
    }

    // /**
    //  * Wakes up all currently awaiting writers and readers.  Their await_resume() function
    //  * will return an expected read result that the ring buffer has stopped.
    //  */
    // auto notify_waiters() -> void
    // {
    //     // Only wake up waiters once.
    //     if (m_stopped.load(std::memory_order::acquire))
    //     {
    //         return;
    //     }

    //     std::unique_lock lk{m_mutex};
    //     m_stopped.exchange(true, std::memory_order::release);

    //     while (m_write_waiters != nullptr)
    //     {
    //         auto* to_resume      = m_write_waiters;
    //         to_resume->m_stopped = true;
    //         m_write_waiters      = m_write_waiters->m_next;

    //         lk.unlock();
    //         to_resume->resume();
    //         lk.lock();
    //     }

    //     while (m_read_waiters != nullptr)
    //     {
    //         auto* to_resume      = m_read_waiters;
    //         to_resume->m_stopped = true;
    //         m_read_waiters       = m_read_waiters->m_next;

    //         lk.unlock();
    //         to_resume->resume();
    //         lk.lock();
    //     }
    // }

  private:
    friend WriteOperation;
    friend ReadOperation;

    mutex_type m_mutex{};

    const std::size_t m_num_elements;
    const SchedulePolicy m_writer_policy;
    const SchedulePolicy m_reader_policy;

    /// The number of items in the ring buffer.
    size_t m_used{0};

    /// The LIFO list of write waiters - single writers will have order perserved
    //  Note: if there are multiple writers order can not be guaranteed, so no need for FIFO
    WriteOperation* m_write_waiters{nullptr};
    /// The LIFO list of read watier.
    ReadOperation* m_read_waiters{nullptr};

    auto try_write_locked(std::unique_lock<mutex_type>& lk) -> bool
    {
        if (m_used == m_num_elements)
        {
            DCHECK(m_read_waiters == nullptr);
            return false;
        }

        ++m_used;

        ReadOperation* to_resume = nullptr;

        if (m_read_waiters != nullptr)
        {
            to_resume      = m_read_waiters;
            m_read_waiters = m_read_waiters->m_next;

            // Since the read operation suspended it needs to be provided an element to read.
            --m_used;  // And we just consumed up another item.
        }

        // release lock
        lk.unlock();

        if (to_resume != nullptr)
        {
            to_resume->resume();
        }

        return true;
    }

    auto try_read_locked(std::unique_lock<mutex_type>& lk, ReadOperation* op) -> bool
    {
        if (m_used == 0)
        {
            return false;
        }

        --m_used;

        WriteOperation* to_resume = nullptr;

        if (m_write_waiters != nullptr)
        {
            to_resume       = m_write_waiters;
            m_write_waiters = m_write_waiters->m_next;

            // Since the write operation suspended it needs to be provided a slot to place its element.
            ++m_used;  // And we just written another item.
        }

        // release lock
        lk.unlock();

        if (to_resume != nullptr)
        {
            to_resume->resume();
        }

        return true;
    }
};

}  // namespace mrc::coroutines
