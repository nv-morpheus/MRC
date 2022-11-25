/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/null_channel.hpp"
#include "mrc/channel/recent_channel.hpp"
#include "mrc/core/userspace_threads.hpp"
#include "mrc/core/watcher.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>  // for sleep_for

#include <chrono>      // for duration, system_clock, milliseconds, time_point
#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t
#include <functional>  // for ref, reference_wrapper
#include <memory>
#include <utility>
// IWYU thinks algorithm is needed for: auto channel = std::make_shared<RecentChannel<int>>(2);
// IWYU pragma: no_include <algorithm>

using namespace mrc;

class TestChannel : public ::testing::Test
{
  protected:
    void SetUp() override {}

    void TearDown() override {}
};

struct TestChannelObserver : public WatcherInterface
{
    ~TestChannelObserver() override = default;

    void on_entry(const WatchableEvent& op, const void* addr) final {}

    void on_exit(const WatchableEvent& op, bool ok, const void* addr) final
    {
        if (ok && op == WatchableEvent::channel_read)
            m_read_counter++;
        if (ok && op == WatchableEvent::channel_write)
            m_write_counter++;
    }

    std::uint64_t m_read_counter{0};
    std::uint64_t m_write_counter{0};
};

TEST_F(TestChannel, FibersBufferedChannelLifeCycle)
{
    std::size_t count = 16;
    boost::fibers::buffered_channel<int> channel(count);

    using status_t = boost::fibers::channel_op_status;

    // channel will block on the count-th element
    for (int i = 0; i < count - 1; i++)
    {
        EXPECT_EQ(channel.try_push(i), status_t::success);
    }

    EXPECT_EQ(channel.try_push(911), status_t::full);

    EXPECT_FALSE(channel.is_closed());
    channel.close();
    EXPECT_TRUE(channel.is_closed());

    // channel can drain remaining elements when closed
    for (int i = 0; i < count - 1; i++)
    {
        int val = -1;
        EXPECT_EQ(channel.pop(val), status_t::success);
    }

    // channel::pop reports closed after last element has been drained
    int val = -1;
    EXPECT_EQ(channel.pop(val), status_t::closed);

    // channel blocks new pushes when closed
    EXPECT_EQ(channel.try_push(911), status_t::closed);
}

TEST_F(TestChannel, NullChannel)
{
    auto null_channel = std::make_shared<NullChannel<int>>();

    channel::Ingress<int>& ingress = *null_channel;
    channel::Egress<int>& egress   = *null_channel;

    ingress.await_write(42);
    ingress.await_write(2);

    auto f = userspace_threads::async([null_channel] {
        boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
        null_channel->close_channel();
    });

    int i;
    auto s = std::chrono::system_clock::now();
    egress.await_read(std::ref(i));
    auto e = std::chrono::system_clock::now();
    auto t = std::chrono::duration<double>(e - s).count();

    EXPECT_GE(t, 0.1);
}

TEST_F(TestChannel, BufferedChannel)
{
    auto channel  = std::make_shared<BufferedChannel<int>>(4);
    auto observer = std::make_shared<TestChannelObserver>();

    channel->add_watcher(observer);

    channel::Ingress<int>& ingress = *channel;
    channel::Egress<int>& egress   = *channel;

    channel->await_write(42);
    ingress.await_write(2);

#ifdef MRC_TRACING_DISABLED
    EXPECT_EQ(observer->m_read_counter, 0);
    EXPECT_EQ(observer->m_write_counter, 0);
#else
    EXPECT_EQ(observer->m_read_counter, 0);
    EXPECT_EQ(observer->m_write_counter, 2);
#endif

    int i;
    egress.await_read(std::ref(i));
    EXPECT_EQ(i, 42);
    egress.try_read(std::ref(i));
    EXPECT_EQ(i, 2);

#ifdef MRC_TRACING_DISABLED
    EXPECT_EQ(observer->m_read_counter, 0);
    EXPECT_EQ(observer->m_write_counter, 0);
#else
    EXPECT_EQ(observer->m_read_counter, 2);
    EXPECT_EQ(observer->m_write_counter, 2);
#endif

    auto f = userspace_threads::async([channel] {
        boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
        channel->close_channel();
    });

    auto s = std::chrono::system_clock::now();
    egress.await_read(std::ref(i));
    auto e = std::chrono::system_clock::now();
    auto t = std::chrono::duration<double>(e - s).count();

    EXPECT_GE(t, 0.1);
}

TEST_F(TestChannel, RecentChannel)
{
    auto channel = std::make_shared<RecentChannel<int>>(2);

    channel::Ingress<int>& ingress = *channel;
    channel::Egress<int>& egress   = *channel;

    channel->await_write(42);
    ingress.await_write(2);
    ingress.await_write(-2);

    int i;
    egress.await_read(std::ref(i));
    EXPECT_EQ(i, 2);
    egress.try_read(std::ref(i));
    EXPECT_EQ(i, -2);

    /*
    auto f = userspace_threads::async([&] {
        boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
        ingress.close_channel();
    });

    auto s = std::chrono::system_clock::now();
    egress.await_read(std::ref(i));
    auto e = std::chrono::system_clock::now();
    auto t = std::chrono::duration<double>(e - s).count();

    EXPECT_GE(t, 0.1);
    */
}

TEST_F(TestChannel, OnComplete) {}

TEST_F(TestChannel, AwaitWriteOverloads)
{
    auto channel = std::make_shared<BufferedChannel<CopyMoveCounter>>(4);

    CopyMoveCounter output;

    auto check_counter =
        [](const CopyMoveCounter& test_val, size_t copy_count, size_t move_count, bool was_copied, bool was_moved) {
            EXPECT_EQ(test_val.copy_count(), copy_count);
            EXPECT_EQ(test_val.move_count(), move_count);
            EXPECT_EQ(test_val.was_copied(), was_copied);
            EXPECT_EQ(test_val.was_moved(), was_moved);
        };

    // === xValue ===
    channel->await_write(CopyMoveCounter());

    channel->await_read(std::ref(output));
    // Old value move in, new value move out
    check_counter(output, 0, 2, false, false);

    // === lValue reference ===
    CopyMoveCounter l_value;
    channel->await_write(l_value);

    channel->await_read(std::ref(output));
    // Old value was copied
    check_counter(l_value, 0, 0, true, false);
    // New value move in and move out
    check_counter(output, 1, 2, false, false);

    // === lValue move ===
    CopyMoveCounter l_value_move;
    channel->await_write(std::move(l_value_move));

    channel->await_read(std::ref(output));
    // Old value was copied
    check_counter(l_value_move, 0, 0, false, true);
    // New value move in and move out
    check_counter(output, 0, 2, false, false);

    // === const lValue ===
    const CopyMoveCounter const_l_value;
    channel->await_write(const_l_value);

    channel->await_read(std::ref(output));
    // Old value was copied
    check_counter(const_l_value, 0, 0, true, false);
    // New value move in and move out
    check_counter(output, 1, 2, false, false);

    // === const rValue ===
    const CopyMoveCounter& const_r_value = CopyMoveCounter();
    channel->await_write(const_r_value);

    channel->await_read(std::ref(output));
    // Old value was copied
    check_counter(const_r_value, 0, 0, true, false);
    // New value move in and move out
    check_counter(output, 1, 2, false, false);
}
