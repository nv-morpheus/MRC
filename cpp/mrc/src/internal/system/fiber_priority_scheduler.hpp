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

#include <boost/fiber/all.hpp>
#include <boost/fiber/scheduler.hpp>

namespace mrc::internal::system {

class FiberPriorityProps : public boost::fibers::fiber_properties
{
  public:
    FiberPriorityProps(boost::fibers::context* ctx) : fiber_properties(ctx), m_priority(0) {}

    int get_priority() const
    {
        return m_priority;
    }

    // Call this method to alter priority, because we must notify
    // priority_scheduler of any change.
    void set_priority(int p)
    {
        // Of course, it's only worth reshuffling the queue and all if we're
        // actually changing the priority.
        if (p != m_priority)
        {
            m_priority = p;
            notify();
        }
    }

  private:
    int m_priority;
};

class FiberPriorityScheduler : public boost::fibers::algo::algorithm_with_properties<FiberPriorityProps>
{
  private:
    using rqueue_t = boost::fibers::scheduler::ready_queue_type;

    rqueue_t m_rqueue;
    std::mutex m_mtx{};
    std::condition_variable m_cnd{};
    bool m_flag{false};

  public:
    FiberPriorityScheduler() : m_rqueue() {}  // NOLINT

    // For a subclass of algorithm_with_properties<>, it's important to
    // override the correct awakened() overload.
    void awakened(boost::fibers::context* ctx, FiberPriorityProps& props) noexcept final
    {
        int ctx_priority = props.get_priority();
        // With this scheduler, fibers with higher priority values are
        // preferred over fibers with lower priority values. But fibers with
        // equal priority values are processed in round-robin fashion. So when
        // we're handed a new context*, put it at the end of the fibers
        // with that same priority. In other words: search for the first fiber
        // in the queue with LOWER priority, and insert before that one.
        rqueue_t::iterator i(
            std::find_if(m_rqueue.begin(), m_rqueue.end(), [this, ctx_priority](boost::fibers::context& c) {
                return properties(&c).get_priority() < ctx_priority;
            }));
        // Now, whether or not we found a fiber with lower priority,
        // insert this new fiber here.
        m_rqueue.insert(i, *ctx);
    }

    boost::fibers::context* pick_next() noexcept final
    {
        // if ready queue is empty, just tell caller
        if (m_rqueue.empty())
        {
            return nullptr;
        }
        boost::fibers::context* ctx(&m_rqueue.front());
        m_rqueue.pop_front();
        return ctx;
    }

    bool has_ready_fibers() const noexcept final
    {
        return !m_rqueue.empty();
    }

    void property_change(boost::fibers::context* ctx, FiberPriorityProps& props) noexcept final
    {
        // Although our priority_props class defines multiple properties, only
        // one of them (priority) actually calls notify() when changed. The
        // point of a property_change() override is to reshuffle the ready
        // queue according to the updated priority value.

        // 'ctx' might not be in our queue at all, if caller is changing the
        // priority of (say) the running fiber. If it's not there, no need to
        // move it: we'll handle it next time it hits awakened().
        if (!ctx->ready_is_linked())
        {
            return;
        }

        // Found ctx: unlink it
        ctx->ready_unlink();

        // Here we know that ctx was in our ready queue, but we've unlinked
        // it. We happen to have a method that will (re-)add a context* to the
        // right place in the ready queue.
        awakened(ctx, props);
    }

    void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept final
    {
        if ((std::chrono::steady_clock::time_point::max)() == time_point)
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cnd.wait(lk, [this]() { return m_flag; });
            m_flag = false;
        }
        else
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cnd.wait_until(lk, time_point, [this]() { return m_flag; });
            m_flag = false;
        }
    }

    void notify() noexcept final
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_flag = true;
        lk.unlock();
        m_cnd.notify_all();
    }
};

}  // namespace mrc::internal::system
