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

#include "mrc/core/thread.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_pool.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <coroutine>
#include <string>

using namespace mrc;

class TestThread : public ::testing::Test
{};

TEST_F(TestThread, GetThreadID)
{
    coroutines::ThreadPool unnamed({.thread_count = 1});
    coroutines::ThreadPool main({.thread_count = 1, .description = "main"});

    auto log_id = [](coroutines::ThreadPool& tp) -> coroutines::Task<std::string> {
        co_await tp.schedule();
        co_return mrc::this_thread::get_id();
    };

    auto from_main    = coroutines::sync_wait(log_id(main));
    auto from_unnamed = coroutines::sync_wait(log_id(unnamed));

    VLOG(1) << mrc::this_thread::get_id();
    VLOG(1) << from_main;
    VLOG(1) << from_unnamed;

    EXPECT_TRUE(mrc::this_thread::get_id().starts_with("sys"));
    EXPECT_TRUE(from_main.starts_with("main"));
    EXPECT_TRUE(from_unnamed.starts_with("thread_pool"));
}
