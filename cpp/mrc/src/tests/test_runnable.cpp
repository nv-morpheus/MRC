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

#include "internal/runnable/resources.hpp"
#include "internal/system/resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/runnable/type_traits.hpp"
#include "mrc/runnable/types.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

using namespace mrc;

#define MRC_DEFAULT_FIBER_PRIORITY 0

static std::shared_ptr<internal::system::System> make_system(std::function<void(Options&)> updater = nullptr)
{
    auto options = std::make_shared<Options>();
    if (updater)
    {
        updater(*options);
    }

    return internal::system::make_system(std::move(options));
}

class TestRunnable : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_system_resources = std::make_unique<internal::system::Resources>(
            internal::system::SystemProvider(make_system([](Options& options) {
                options.topology().user_cpuset("0-3");
                options.topology().restrict_gpus(true);
                options.engine_factories().set_engine_factory_options("thread_pool", [](EngineFactoryOptions& options) {
                    options.engine_type   = runnable::EngineType::Thread;
                    options.allow_overlap = false;
                    options.cpu_count     = 2;
                });
            })));

        m_resources = std::make_unique<internal::runnable::Resources>(*m_system_resources, 0);
    }

    void TearDown() override
    {
        m_resources.reset();
        m_system_resources.reset();
    }

    std::unique_ptr<internal::system::Resources> m_system_resources;
    std::unique_ptr<internal::runnable::Resources> m_resources;
};

class TestGenericRunnable final : public runnable::RunnableWithContext<>
{
    void run(ContextType& ctx) final
    {
        LOG(INFO) << info(ctx) << ": do_run begin";
        auto status = state();
        while (status == State::Run)
        {
            boost::this_fiber::sleep_for(std::chrono::milliseconds(50));
            status = state();
        }
        LOG(INFO) << info(ctx) << ": do_run end";
    }
};

class TestFiberRunnable final : public runnable::FiberRunnable<>
{
    void run(ContextType& ctx) final
    {
        LOG(INFO) << info(ctx) << ": do_run begin";
        auto status = state();
        while (status == State::Run)
        {
            boost::this_fiber::sleep_for(std::chrono::milliseconds(50));
            status = state();
        }
        LOG(INFO) << info(ctx) << ": do_run end";
    }

  public:
    int i;
};

class TestThreadRunnable final : public runnable::ThreadRunnable<>
{
    void run(ContextType& ctx) final
    {
        LOG(INFO) << info(ctx) << ": do_run begin";
        auto status = state();
        while (status == State::Run)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            status = state();
        }
        LOG(INFO) << info(ctx) << ": do_run end";
    }
};

TEST_F(TestRunnable, TypeTraitsGeneric)
{
    using ctx_t = runnable::runnable_context_t<TestGenericRunnable>;
    static_assert(std::is_same_v<ctx_t, runnable::Context>, "should be true");
    static_assert(runnable::is_unwrapped_context_v<ctx_t>, "should be true");
    static_assert(!runnable::is_fiber_runnable_v<TestGenericRunnable>, "should be false");
    static_assert(!runnable::is_fiber_context_v<ctx_t>, "should be false");
    static_assert(!runnable::is_thread_context_v<ctx_t>, "should be false");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestRunnable, TypeTraitsFiber)
{
    using ctx_t = runnable::runnable_context_t<TestFiberRunnable>;
    static_assert(runnable::is_fiber_runnable_v<TestFiberRunnable>, "should be true");
    static_assert(runnable::is_fiber_context_v<ctx_t>, "should be true");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestRunnable, TypeTraitsThread)
{
    using ctx_t = runnable::runnable_context_t<TestThreadRunnable>;
    static_assert(!runnable::is_fiber_runnable_v<TestThreadRunnable>, "should be false");
    static_assert(runnable::is_thread_context_v<ctx_t>, "should be true");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestRunnable, GenericRunnableRunWithFiber)
{
    runnable::LaunchOptions main;
    std::atomic<std::size_t> counter = 0;

    main.engine_factory_name = "default";  // running this on main would fail, since main can only have pe_count == 1
    main.pe_count            = 2;
    main.engines_per_pe      = 1;

    auto runnable = std::make_unique<TestGenericRunnable>();
    auto launcher = m_resources->launch_control().prepare_launcher(main, std::move(runnable));

    launcher->apply([&counter](runnable::Runner& runner) {
        runner.on_instance_state_change_callback([&counter](const runnable::Runnable& runnable,
                                                            std::size_t id,
                                                            runnable::Runner::State old_state,
                                                            runnable::Runner::State new_state) { ++counter; });
    });

    auto runner = launcher->ignition();

    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();

    // counter should be 3 for each instance
    // Queued, Running, Completed
    EXPECT_EQ(counter, 3 * main.pe_count * main.engines_per_pe);
}

TEST_F(TestRunnable, GenericRunnableRunWithLaunchControl)
{
    auto runnable = std::make_unique<TestGenericRunnable>();

    auto runner = m_resources->launch_control().prepare_launcher(std::move(runnable))->ignition();

    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestRunnable, GenericRunnableRunWithThread)
{
    runnable::LaunchOptions thread_pool;
    thread_pool.engine_factory_name = "thread_pool";
    thread_pool.pe_count            = 2;

    auto runnable = std::make_unique<TestGenericRunnable>();
    auto runner   = m_resources->launch_control().prepare_launcher(thread_pool, std::move(runnable))->ignition();

    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestRunnable, FiberRunnable)
{
    runnable::LaunchOptions factory;
    factory.engine_factory_name = "default";
    factory.pe_count            = 2;

    auto runnable = std::make_unique<TestFiberRunnable>();
    auto runner   = m_resources->launch_control().prepare_launcher(factory, std::move(runnable))->ignition();

    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestRunnable, FiberRunnableMisMatch)
{
    runnable::LaunchOptions factory;
    factory.engine_factory_name = "thread_pool";
    factory.pe_count            = 2;

    auto runnable = std::make_unique<TestFiberRunnable>();

    // todo(#188) - this should throw an exception rather than abort
    EXPECT_DEATH(auto launcher = m_resources->launch_control().prepare_launcher(factory, std::move(runnable)), "");
}

TEST_F(TestRunnable, RunnerOutOfScope)
{
    runnable::LaunchOptions factory;
    factory.engine_factory_name = "default";
    factory.pe_count            = 2;

    auto runnable = std::make_unique<TestFiberRunnable>();
    auto runner   = m_resources->launch_control().prepare_launcher(factory, std::move(runnable))->ignition();

    runner->await_live();
}

// Move the remaining tests to TestNode

// TEST_F(TestRunnable, ThreadRunnable)
// {
//     CpuSet cpus("0,1");

//     auto runnable = std::make_unique<TestThreadRunnable>();
//     auto runner   = runnable::make_runner(std::move(runnable));
//     auto launcher = std::make_shared<runnable::ThreadEngines>(cpus, m_topology);

//     runner->enqueue(launcher);
//     runner->await_live();
//     std::this_thread::sleep_for(std::chrono::milliseconds(50));
//     runner->stop();
//     runner->await_join();
// }

// TEST_F(TestRunnable, OperatorMuxer)
// {
//     std::atomic<std::size_t> counter = 0;
//     std::unique_ptr<runnable::Runner> runner_source;
//     std::unique_ptr<runnable::Runner> runner_sink;

//     // do the construction in its own scope
//     // only allow the runners to escape the scope
//     // this ensures that the Muxer Operator survives
//     {
//         auto source =
//             std::make_unique<node::RxSource<float>>(rxcpp::observable<>::create<float>([](rxcpp::subscriber<float> s)
//             {
//                 s.on_next(1.0f);
//                 s.on_next(2.0f);
//                 s.on_next(3.0f);
//                 s.on_completed();
//             }));
//         auto muxer = std::make_shared<node::Muxer<float>>();
//         auto sink =
//             std::make_unique<node::RxSink<float>>(rxcpp::make_observer_dynamic<float>([&](float x) { ++counter; }));

//         node::make_edge(*source, *muxer);
//         node::make_edge(*muxer, *sink);

//         runner_sink   = m_launch_control->prepare_launcher(std::move(sink))->ignition();
//         runner_source = m_launch_control->prepare_launcher(std::move(source))->ignition();
//     }

//     runner_source->await_join();
//     runner_sink->await_join();

//     EXPECT_EQ(counter, 3);
// }

// TEST_F(TestRunnable, IdentityNode)
// {
//     std::atomic<std::size_t> counter = 0;
//     std::unique_ptr<runnable::Runner> runner_source;
//     std::unique_ptr<runnable::Runner> runner_passthru;
//     std::unique_ptr<runnable::Runner> runner_sink;

//     // do the construction in its own scope
//     // only allow the runners to escape the scope
//     // this ensures that the Muxer Operator survives
//     {
//         auto source =
//             std::make_unique<node::RxSource<float>>(rxcpp::observable<>::create<float>([](rxcpp::subscriber<float> s)
//             {
//                 s.on_next(1.0f);
//                 s.on_next(2.0f);
//                 s.on_next(3.0f);
//                 s.on_completed();
//             }));
//         auto passthru = std::make_unique<node::RxNode<float>>(
//             rxcpp::operators::tap([this](const float& t) { LOG(INFO) << "tap = " << t; }));
//         auto sink =
//             std::make_unique<node::RxSink<float>>(rxcpp::make_observer_dynamic<float>([&](float x) { ++counter; }));

//         node::make_edge(*source, *passthru);
//         node::make_edge(*passthru, *sink);

//         runner_sink     = m_launch_control->prepare_launcher(std::move(sink))->ignition();
//         runner_passthru = m_launch_control->prepare_launcher(std::move(passthru))->ignition();
//         runner_source   = m_launch_control->prepare_launcher(std::move(source))->ignition();
//     }

//     runner_source->await_join();
//     runner_passthru->await_join();
//     runner_sink->await_join();

//     EXPECT_EQ(counter, 3);
// }
