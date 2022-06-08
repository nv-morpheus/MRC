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

#include <srf/options/topology.hpp>
#include <srf/types.hpp>
#include "internal/system/fiber_pool.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/system.hpp"
#include "internal/system/thread_pool.hpp"
#include "internal/system/topology.hpp"
#include "srf/core/bitmap.hpp"
#include "srf/core/fiber_pool.hpp"
#include "srf/options/options.hpp"
#include "srf/utils/thread_local_shared_pointer.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <set>
#include <thread>
#include <vector>

using namespace srf;
using namespace internal;

using system::System;
using system::ThreadPool;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestSystem : public ::testing::Test
{
  protected:
    static std::shared_ptr<Options> make_options(std::function<void(Options&)> updater = nullptr)
    {
        auto options = std::make_shared<Options>();
        if (updater)
        {
            updater(*options);
        }

        return options;
    }
};

TEST_F(TestSystem, LifeCycle)
{
    auto system = System::make_system(make_options());
}

TEST_F(TestSystem, FiberPool)
{
    auto system = System::make_system(make_options([](Options& options) {
        // ensure we have 4 logical cpus
        options.topology().user_cpuset("0-3");
    }));

    CpuSet cpu_set;
    cpu_set.on(0);
    cpu_set.on(2);
    EXPECT_EQ(cpu_set.weight(), 2);

    auto pool = system->make_fiber_pool(cpu_set);

    EXPECT_EQ(pool->thread_count(), 2);

    // now we need to ensure that all fibers on each thread
    // stay pinned in their respective threads

    std::vector<Future<std::thread::id>> t0;
    std::vector<Future<std::thread::id>> t1;

    // issue 10 fibers which each issue 10 fibers
    // ensure all fibers yield at least once and report
    // back the same thread_id
    // there are two threads with fiber schedulers enabled
    // so this test provides some evidence that the fibers
    // are not migrating.
    for (int i = 0; i < 10; i++)
    {
        t0.push_back(pool->enqueue(0, [] {
            std::vector<Future<std::thread::id>> fs;
            for (int j = 0; j < 10; j++)
            {
                fs.push_back(boost::fibers::async([] {
                    boost::this_fiber::sleep_for(std::chrono::milliseconds(50));
                    return std::this_thread::get_id();
                }));
            }
            boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
            for (auto& f : fs)
            {
                auto id = f.get();
                CHECK_EQ(id, std::this_thread::get_id());
            }
            boost::this_fiber::yield();
            return std::this_thread::get_id();
        }));
    }

    std::set<std::thread::id> s0;

    for (int i = 0; i < 10; i++)
    {
        s0.insert(t0[i].get());
    }

    EXPECT_EQ(s0.size(), 1);
}

TEST_F(TestSystem, ImpossibleCoreCount)
{
    auto system = System::make_system(make_options([](Options& options) {
        // ensure we have 2 logical cpus
        options.topology().user_cpuset("0,1");
    }));

    CpuSet cpu_set;
    cpu_set.on(0);
    cpu_set.on(99999999);
    EXPECT_EQ(cpu_set.weight(), 2);

    EXPECT_ANY_THROW(auto pool = system->make_fiber_pool(cpu_set));
}

TEST_F(TestSystem, RoundRobinFiberPool)
{
    auto system = System::make_system(make_options([](Options& options) {
        // ensure we have 2 logical cpus
        options.topology().user_cpuset("0,1,2");
    }));

    if (system->topology().cpu_set().weight() < 3)
    {
        GTEST_SKIP() << "At least three threads are required to test the RoundRobinFiberPool";
    }

    CpuSet cpu_set;
    cpu_set.on(0);
    cpu_set.on(2);
    EXPECT_EQ(cpu_set.weight(), 2);

    auto pool = std::make_shared<core::RoundRobinFiberPool>(system->make_fiber_pool(cpu_set));

    std::vector<Future<std::uint32_t>> futures;

    for (int i = 0; i < 10; i++)
    {
        futures.push_back(pool->enqueue([system] {
            auto cpu_set = system->get_current_thread_affinity();
            EXPECT_EQ(cpu_set.weight(), 1);
            return cpu_set.first();
        }));
    }

    for (int i = 0; i < 10; i++)
    {
        auto& f = futures.at(i);
        if (i % 2 == 0)
        {
            EXPECT_EQ(f.get(), 0);
        }
        else
        {
            EXPECT_EQ(f.get(), 2);
        }
    }
}

TEST_F(TestSystem, ThreadLocalResource)
{
    auto system = System::make_system(make_options());

    auto pool0 = system->make_fiber_pool(CpuSet("0,1"));
    auto pool1 = system->make_fiber_pool(CpuSet("2,3"));

    auto i0 = std::make_shared<int>(0);
    auto i1 = std::make_shared<int>(2);

    pool0->enqueue(0, [] { EXPECT_ANY_THROW(utils::ThreadLocalSharedPointer<int>::get()); }).get();
    pool0->enqueue(1, [] { EXPECT_ANY_THROW(utils::ThreadLocalSharedPointer<int>::get()); }).get();
    pool1->enqueue(0, [] { EXPECT_ANY_THROW(utils::ThreadLocalSharedPointer<int>::get()); }).get();
    pool1->enqueue(1, [] { EXPECT_ANY_THROW(utils::ThreadLocalSharedPointer<int>::get()); }).get();

    pool0->set_thread_local_resource(i0);
    pool1->set_thread_local_resource(i1);

    auto j0 = pool0->enqueue(0, [] { return *utils::ThreadLocalSharedPointer<int>::get(); }).get();
    auto j1 = pool1->enqueue(0, [] { return *utils::ThreadLocalSharedPointer<int>::get(); }).get();

    EXPECT_EQ(j0, 0);
    EXPECT_EQ(j1, 2);
}

TEST_F(TestSystem, ThreadInitializersAndFinalizers)
{
    auto system = System::make_system(make_options([](Options& options) {
        options.topology().user_cpuset("0-1");
        options.topology().restrict_gpus(true);
    }));

    std::atomic<std::size_t> init_counter = 0;
    std::atomic<std::size_t> fini_counter = 0;

    system->register_thread_local_initializer(system->topology().cpu_set(), [&init_counter] { init_counter++; });

    EXPECT_EQ(init_counter, 2);
    EXPECT_EQ(fini_counter, 0);

    system->register_thread_local_finalizer(system->topology().cpu_set(), [&fini_counter] { fini_counter++; });

    system->make_thread(CpuSet("0"), [] { VLOG(10) << "test thread"; }).join();

    EXPECT_EQ(init_counter, 3);
    EXPECT_EQ(fini_counter, 1);

    system.reset();

    EXPECT_EQ(init_counter, 3);
    EXPECT_EQ(fini_counter, 3);
}

TEST_F(TestSystem, ThreadPool)
{
    auto system = System::make_system(make_options([](Options& options) {
        options.topology().user_cpuset("0-3");
        options.topology().restrict_gpus(true);
    }));

    std::atomic<std::size_t> counter = 0;

    auto thread_pool = std::make_unique<ThreadPool>(system, CpuSet("2-3"), 2);

    auto f = [&counter] {
        ++counter;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return std::this_thread::get_id();
    };

    std::vector<Future<std::thread::id>> futures;
    std::set<std::thread::id> ids;

    futures.push_back(thread_pool->enqueue(f));
    futures.push_back(thread_pool->enqueue(f));
    futures.push_back(thread_pool->enqueue(f));

    for (auto& f : futures)
    {
        ids.insert(f.get());
    }

    EXPECT_EQ(counter, 3);
    EXPECT_EQ(ids.size(), 2);
}
