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

#include "./test_srf.hpp"  // IWYU pragma: associated

#include <srf/core/bitmap.hpp>
#include <srf/core/fiber_pool.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/forward.hpp>
#include <srf/node/operators/muxer.hpp>
#include <srf/node/operators/operator.hpp>
#include <srf/node/rx_node.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/rx_source.hpp>
#include <srf/node/source_channel.hpp>
#include <srf/options/fiber_pool.hpp>
#include <srf/options/topology.hpp>
#include <srf/runnable/context.hpp>
#include <srf/runnable/engine.hpp>
#include <srf/runnable/forward.hpp>
#include <srf/runnable/launch_control.hpp>
#include <srf/runnable/runner.hpp>
#include <srf/runnable/type_traits.hpp>
#include "internal/system/topology.hpp"

#include <gtest/gtest.h>
#include <boost/fiber/operations.hpp>

#include <chrono>
#include <memory>
#include <sstream>
#include <string>  // for string operator<<
#include <thread>
#include <type_traits>
#include <utility>  // for move

#define SRF_DEFAULT_FIBER_PRIORITY 0

class TestCore : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_topology_options.user_cpuset("0-3");
        m_topology       = Topology::Create(m_topology_options);
        m_fiber_pool_mgr = FiberPoolManager::Create(m_topology);

        auto memory_resources = std::make_shared<TestCoreResorucesImpl>();
        utils::ThreadLocalSharedPointer<core::Resources>::set(memory_resources);

        // set up launch control - this is what the scheduler will do

        runnable::LaunchControlConfig config;

        // divide the cpu from the topology up into pool
        auto default_pool = m_fiber_pool_mgr->make_pool(CpuSet("0-2"));
        auto mgmt_pool    = m_fiber_pool_mgr->make_pool(CpuSet("3"));

        // set thread local data on each thread in the fiber pool
        default_pool->set_thread_local_resource(memory_resources);
        mgmt_pool->set_thread_local_resource(memory_resources);

        // these are the actual object that LaunchControl needs
        auto default_resources = std::make_shared<runnable::ReusableFiberEngineFactory>(default_pool);
        auto mgmt_resources    = std::make_shared<runnable::ReusableFiberEngineFactory>(mgmt_pool);

        config.resource_groups[default_engine_factory_name()] = default_resources;
        config.resource_groups["mgmt"]                        = mgmt_resources;

        runnable::LaunchOptions service_defaults;
        service_defaults.engine_factory_name = "mgmt";
        config.services.set_default_options(service_defaults);

        m_launch_control = std::make_shared<runnable::LaunchControl>(std::move(config));
    }

    void TearDown() override {}

    TopologyOptions m_topology_options;
    std::shared_ptr<FiberPoolManager> m_fiber_pool_mgr;
    std::shared_ptr<Topology> m_topology;
    std::shared_ptr<runnable::LaunchControl> m_launch_control;
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

TEST_F(TestCore, TypeTraitsGeneric)
{
    using ctx_t = runnable::runnable_context_t<TestGenericRunnable>;
    static_assert(std::is_same_v<ctx_t, runnable::Context>, "should be true");
    static_assert(runnable::is_unwrapped_context_v<ctx_t>, "should be true");
    static_assert(!runnable::is_fiber_runnable_v<TestGenericRunnable>, "should be false");
    static_assert(!runnable::is_fiber_context_v<ctx_t>, "should be false");
    static_assert(!runnable::is_thread_context_v<ctx_t>, "should be false");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestCore, TypeTraitsFiber)
{
    using ctx_t = runnable::runnable_context_t<TestFiberRunnable>;
    static_assert(runnable::is_fiber_runnable_v<TestFiberRunnable>, "should be true");
    static_assert(runnable::is_fiber_context_v<ctx_t>, "should be true");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestCore, TypeTraitsThread)
{
    using ctx_t = runnable::runnable_context_t<TestThreadRunnable>;
    static_assert(!runnable::is_fiber_runnable_v<TestThreadRunnable>, "should be false");
    static_assert(runnable::is_thread_context_v<ctx_t>, "should be true");
    static_assert(std::is_same_v<runnable::unwrap_context_t<ctx_t>, runnable::Context>, "true");
}

TEST_F(TestCore, GenericRunnableRunWithFiber)
{
    EXPECT_EQ(m_fiber_pool_mgr->thread_count(), 4);
    auto pool = m_fiber_pool_mgr->make_pool(CpuSet("0,1"));

    auto runnable = std::make_unique<TestGenericRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::FiberEngines>(pool);

    std::size_t counter = 0;
    runner->on_state_change_callback([&counter](const runnable::Runnable& runnable,
                                                std::size_t id,
                                                runnable::Runner::State old_state,
                                                runnable::Runner::State new_state) { ++counter; });

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();

    // counter should be 3 for each instance
    // Queued, Running, Completed
    EXPECT_EQ(counter, 3 * pool->thread_count());
}

TEST_F(TestCore, GenericRunnableRunWithLaunchControl)
{
    auto runnable = std::make_unique<TestGenericRunnable>();

    auto runner = m_launch_control->prepare_launcher(std::move(runnable))->ignition();

    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestCore, GenericRunnableRunWithThread)
{
    CpuSet cpus("0,1");

    auto runnable = std::make_unique<TestGenericRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::ThreadEngines>(cpus, m_topology);

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestCore, FiberRunnable)
{
    EXPECT_EQ(m_fiber_pool_mgr->thread_count(), 4);
    auto pool = m_fiber_pool_mgr->make_pool(CpuSet("0,1"));

    auto runnable = std::make_unique<TestFiberRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::FiberEngines>(pool);

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestCore, FiberRunnableWithLaunchControl)
{
    std::map<runnable::Runner::State, std::atomic<std::size_t>> counters;
    auto event_sink = std::make_unique<node::RxSink<runnable::RunnerEvent>>(
        [&counters](runnable::RunnerEvent event) { counters[event.new_state]++; });

    auto runnable = std::make_unique<TestFiberRunnable>();
    auto launcher = m_launch_control->prepare_launcher(std::move(runnable));

    GTEST_SKIP() << "fix test to use lambda";

    // launcher->set_runner_callback([&counters](const runnable::Runnable& runnable,
    //                                           std::size_t id,
    //                                           runnable::Runner::State old_state,
    //                                           runnable::Runner::State new_state) { counters[new_state]++; });

    // launcher is enabled
    auto runner = launcher->ignition();
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();

    // we miss the completed event for event_sink since the release of launch_control
    // which closes the sink channel before the complete event can be emitted on the channel
    EXPECT_EQ(counters[runnable::Runner::State::Queued], 1);
    EXPECT_EQ(counters[runnable::Runner::State::Running], 1);
    EXPECT_EQ(counters[runnable::Runner::State::Completed], 1);
}

TEST_F(TestCore, ThreadRunnable)
{
    CpuSet cpus("0,1");

    auto runnable = std::make_unique<TestThreadRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::ThreadEngines>(cpus, m_topology);

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestCore, ThreadRunnableWithLaunchControl)
{
    GTEST_SKIP() << "Launch Control is not yet configured for ThreadEngines";

    CpuSet cpus("0,1");

    auto runnable = std::make_unique<TestThreadRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::ThreadEngines>(cpus, m_topology);

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    runner->stop();
    runner->await_join();
}

TEST_F(TestCore, RunnerOutOfScope)
{
    EXPECT_EQ(m_fiber_pool_mgr->thread_count(), 4);

    auto pool = m_fiber_pool_mgr->make_pool(CpuSet("0-1"));
    EXPECT_EQ(pool->thread_count(), 2);

    auto runnable = std::make_unique<TestFiberRunnable>();
    auto runner   = runnable::make_runner(std::move(runnable));
    auto launcher = std::make_shared<runnable::FiberEngines>(pool);

    runner->enqueue(launcher);
    runner->await_live();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_F(TestCore, OperatorMuxer)
{
    std::atomic<std::size_t> counter = 0;
    std::unique_ptr<runnable::Runner> runner_source;
    std::unique_ptr<runnable::Runner> runner_sink;

    // do the construction in its own scope
    // only allow the runners to escape the scope
    // this ensures that the Muxer Operator survives
    {
        auto source =
            std::make_unique<node::RxSource<float>>(rxcpp::observable<>::create<float>([](rxcpp::subscriber<float> s) {
                s.on_next(1.0f);
                s.on_next(2.0f);
                s.on_next(3.0f);
                s.on_completed();
            }));
        auto muxer = std::make_shared<node::Muxer<float>>();
        auto sink =
            std::make_unique<node::RxSink<float>>(rxcpp::make_observer_dynamic<float>([&](float x) { ++counter; }));

        node::make_edge(*source, *muxer);
        node::make_edge(*muxer, *sink);

        runner_sink   = m_launch_control->prepare_launcher(std::move(sink))->ignition();
        runner_source = m_launch_control->prepare_launcher(std::move(source))->ignition();
    }

    runner_source->await_join();
    runner_sink->await_join();

    EXPECT_EQ(counter, 3);
}

TEST_F(TestCore, IdentityNode)
{
    std::atomic<std::size_t> counter = 0;
    std::unique_ptr<runnable::Runner> runner_source;
    std::unique_ptr<runnable::Runner> runner_passthru;
    std::unique_ptr<runnable::Runner> runner_sink;

    // do the construction in its own scope
    // only allow the runners to escape the scope
    // this ensures that the Muxer Operator survives
    {
        auto source =
            std::make_unique<node::RxSource<float>>(rxcpp::observable<>::create<float>([](rxcpp::subscriber<float> s) {
                s.on_next(1.0f);
                s.on_next(2.0f);
                s.on_next(3.0f);
                s.on_completed();
            }));
        auto passthru = std::make_unique<node::RxNode<float>>(
            rxcpp::operators::tap([this](const float& t) { LOG(INFO) << "tap = " << t; }));
        auto sink =
            std::make_unique<node::RxSink<float>>(rxcpp::make_observer_dynamic<float>([&](float x) { ++counter; }));

        node::make_edge(*source, *passthru);
        node::make_edge(*passthru, *sink);

        runner_sink     = m_launch_control->prepare_launcher(std::move(sink))->ignition();
        runner_passthru = m_launch_control->prepare_launcher(std::move(passthru))->ignition();
        runner_source   = m_launch_control->prepare_launcher(std::move(source))->ignition();
    }

    runner_source->await_join();
    runner_passthru->await_join();
    runner_sink->await_join();

    EXPECT_EQ(counter, 3);
}
