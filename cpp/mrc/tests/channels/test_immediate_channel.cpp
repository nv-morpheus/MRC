/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/v2/immediate_channel.hpp"
#include "mrc/coroutines/latch.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace mrc;
using namespace mrc::channel;
using namespace mrc::channel::v2;

class TestChannelV2 : public ::testing::Test
{
  protected:
    void SetUp() override {}

    void TearDown() override {}

    ImmediateChannel<int> m_channel;

    coroutines::Task<void> int_writer(int iterations, coroutines::Latch& latch)
    {
        for (int i = 0; i < iterations; i++)
        {
            co_await m_channel.async_write(std::move(i));
        }
        latch.count_down();
        co_return;
    }

    coroutines::Task<> close_on_latch(coroutines::Latch& latch)
    {
        co_await latch;
        m_channel.close();
        co_return;
    }

    coroutines::Task<void> int_reader(int iterations)
    {
        int i = 0;
        while (true)
        {
            auto data = co_await m_channel.async_read();
            if (!data)
            {
                break;
            }
            i++;
        }
        EXPECT_EQ(i, iterations);
        co_return;
    }
};

TEST_F(TestChannelV2, ChannelClosed)
{
    ImmediateChannel<int> channel;
    channel.close();

    auto test = [&]() -> coroutines::Task<void> {
        // write should throw
        EXPECT_ANY_THROW(co_await channel.async_write(42));

        // read should return unexpected
        auto data = co_await channel.async_read();
        EXPECT_FALSE(data);

        // task throws
        co_await channel.async_write(42);
        co_return;
    };

    EXPECT_ANY_THROW(coroutines::sync_wait(test()));
}

TEST_F(TestChannelV2, SingleWriterSingleReader)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(coroutines::when_all(close_on_latch(latch), int_writer(3, latch), int_reader(3)));
}

TEST_F(TestChannelV2, Readerx1_Writer_x1)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(coroutines::when_all(int_reader(3), int_writer(3, latch), close_on_latch(latch)));
}

TEST_F(TestChannelV2, Readerx2_Writer_x1)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(
        coroutines::when_all(int_reader(2), int_reader(1), int_writer(3, latch), close_on_latch(latch)));
}

TEST_F(TestChannelV2, Readerx3_Writer_x1)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(
        coroutines::when_all(close_on_latch(latch), int_reader(1), int_reader(1), int_reader(1), int_writer(3, latch)));
}

TEST_F(TestChannelV2, Readerx4_Writer_x1)
{
    // reader are a lifo, so the first reader in the task list will not get a data entry
    coroutines::Latch latch{1};
    coroutines::sync_wait(coroutines::when_all(
        close_on_latch(latch), int_reader(0), int_reader(1), int_reader(1), int_reader(1), int_writer(3, latch)));
}

TEST_F(TestChannelV2, Readerx3_Writer_x1_Reader_x1)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(coroutines::when_all(
        int_reader(1), int_reader(1), close_on_latch(latch), int_reader(1), int_writer(3, latch), int_reader(0)));
}

TEST_F(TestChannelV2, Writer_2_Reader_x2)
{
    coroutines::Latch latch{1};
    coroutines::sync_wait(coroutines::when_all(
        int_writer(2, latch), int_writer(2, latch), close_on_latch(latch), int_reader(4), int_reader(0)));
}
