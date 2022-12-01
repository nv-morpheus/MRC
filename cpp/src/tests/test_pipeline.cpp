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

#include "nodes/common_nodes.hpp"
#include "pipelines/common_pipelines.hpp"

#include "internal/pipeline/manager.hpp"
#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/topology.hpp"
#include "internal/utils/collision_detector.hpp"

#include "mrc/channel/channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/addresses.hpp"
#include "mrc/core/executor.hpp"
#include "mrc/data/reusable_pool.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/hana/if.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace mrc;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestPipeline : public ::testing::Test
{};

static std::shared_ptr<internal::system::System> make_system(std::function<void(Options&)> updater = nullptr)
{
    auto options = std::make_shared<Options>();
    if (updater)
    {
        updater(*options);
    }

    return internal::system::make_system(std::move(options));
}

static std::shared_ptr<internal::pipeline::Pipeline> unwrap(internal::pipeline::IPipeline& pipeline)
{
    return internal::pipeline::Pipeline::unwrap(pipeline);
}

static void run_custom_manager(std::unique_ptr<internal::pipeline::IPipeline> pipeline,
                               internal::pipeline::SegmentAddresses&& update,
                               bool delayed_stop = false)
{
    auto resources = internal::resources::Manager(internal::system::SystemProvider(make_system([](Options& options) {
        options.topology().user_cpuset("0-1");
        options.topology().restrict_gpus(true);
    })));

    auto manager = std::make_unique<internal::pipeline::Manager>(unwrap(*pipeline), resources);

    auto f = std::async([&] {
        if (delayed_stop)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            manager->service_stop();
        }
    });

    manager->service_start();
    manager->push_updates(std::move(update));
    manager->service_await_join();

    f.get();
}

static void run_manager(std::unique_ptr<internal::pipeline::IPipeline> pipeline, bool delayed_stop = false)
{
    auto resources = internal::resources::Manager(internal::system::SystemProvider(make_system([](Options& options) {
        options.topology().user_cpuset("0");
        options.topology().restrict_gpus(true);
        mrc::channel::set_default_channel_size(64);
    })));

    auto manager = std::make_unique<internal::pipeline::Manager>(unwrap(*pipeline), resources);

    internal::pipeline::SegmentAddresses update;
    update[segment_address_encode(segment_name_hash("seg_1"), 0)] = 0;

    auto f = std::async([&] {
        if (delayed_stop)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            manager->service_stop();
        }
    });

    manager->service_start();
    manager->push_updates(std::move(update));
    manager->service_await_join();

    f.get();
}

TEST_F(TestPipeline, PortNamingService)
{
    internal::utils::CollisionDetector hasher;

    auto p1 = hasher.register_name("test 1");
    auto p2 = hasher.register_name("test 2");
    EXPECT_NE(p1, p2);

    auto p3 = hasher.register_name("test 1");
    EXPECT_EQ(p1, p3);
}

TEST_F(TestPipeline, LifeCycle)
{
    run_manager(test::pipelines::finite_single_segment());
}

TEST_F(TestPipeline, LifeCycleWithException)
{
    EXPECT_ANY_THROW(run_manager(test::pipelines::finite_single_segment_will_throw()));
}

TEST_F(TestPipeline, LifeCycleWithExceptionAndInfiniteSource)
{
    auto pipeline = pipeline::make_pipeline();

    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto rx_source = s.make_object("rx_source", test::nodes::infinite_int_rx_source());
        auto rx_sink   = s.make_object("rx_sink", test::nodes::int_sink_throw_on_even());
        s.make_edge(rx_source, rx_sink);
    });

    EXPECT_ANY_THROW(run_manager(std::move(pipeline)));
}

TEST_F(TestPipeline, LifeCycleStop)
{
    auto pipeline = pipeline::make_pipeline();

    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto rx_source = s.make_object("rx_source", test::nodes::infinite_int_rx_source());
        auto rx_sink   = s.make_object("rx_sink", test::nodes::int_sink());
        s.make_edge(rx_source, rx_sink);
    });

    run_manager(std::move(pipeline), true);
}

TEST_F(TestPipeline, Queue)
{
    auto pipeline = pipeline::make_pipeline();

    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto source = s.make_object("source", test::nodes::infinite_int_rx_source());
        auto queue  = s.make_object("queue", std::make_unique<node::Queue<int>>());
        auto sink   = s.make_object("sink", test::nodes::int_sink());
        s.make_edge(source, queue);
        s.make_edge(queue, sink);
    });

    run_manager(std::move(pipeline), true);
}

TEST_F(TestPipeline, InitializerThrows)
{
    auto pipeline = pipeline::make_pipeline();
    auto segment  = pipeline->make_segment("seg_1", [](segment::Builder& s) { throw std::runtime_error("no bueno"); });
    EXPECT_ANY_THROW(run_manager(std::move(pipeline)));
}

TEST_F(TestPipeline, DuplicateNameInSegment)
{
    auto pipeline = pipeline::make_pipeline();

    // this should fail to register the int_sink because of a duplicate name
    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto rx_source = s.make_object("rx_source", test::nodes::finite_int_rx_source());
        auto rx_sink   = s.make_object("rx_source", test::nodes::int_sink());
        s.make_edge(rx_source, rx_sink);
    });

    EXPECT_DEATH(run_manager(std::move(pipeline)), "");
}

TEST_F(TestPipeline, ExecutorLifeCycle)
{
    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(test::pipelines::finite_single_segment());
    executor.start();
    executor.join();
}

TEST_F(TestPipeline, MultiSegment)
{
    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(test::pipelines::finite_multisegment());
    executor.start();
    executor.join();
}

TEST_F(TestPipeline, MultiSegmentLoadBalancer)
{
    // the default connection/manifold type between segments is a load balancer
    // this test we create one copy of our source segment (seg_1) and two copies of our sink segment (seg_2)
    // we collect the fiber id for the sink runnable processing each data element,
    // then we count the unique fiber ids collected

    auto pipeline = pipeline::make_pipeline();

    int count = 1000;
    std::mutex mutex;
    std::vector<boost::fibers::fiber::id> ranks;

    pipeline->make_segment("seg_1", segment::EgressPorts<int>({"i"}), [count](segment::Builder& s) {
        auto src    = s.make_object("src", test::nodes::finite_int_rx_source(count));
        auto egress = s.get_egress<int>("i");
        s.make_edge(src, egress);
    });

    pipeline->make_segment("seg_2", segment::IngressPorts<int>({"i"}), [&mutex, &ranks](segment::Builder& s) mutable {
        auto sink    = s.make_sink<int>("sink", [&](int x) {
            VLOG(1) << runnable::Context::get_runtime_context().info() << ": data=" << x;
            std::lock_guard<decltype(mutex)> lock(mutex);
            ranks.push_back(boost::this_fiber::get_id());
        });
        auto ingress = s.get_ingress<int>("i");
        s.make_edge(ingress, sink);
    });

    // run 1 copy of seg_1 and 2 copies of seg_2 all on parition 0
    internal::pipeline::SegmentAddresses update;
    update[segment_address_encode(segment_name_hash("seg_1"), 0)] = 0;
    update[segment_address_encode(segment_name_hash("seg_2"), 0)] = 0;
    update[segment_address_encode(segment_name_hash("seg_2"), 1)] = 0;

    run_custom_manager(std::move(pipeline), std::move(update));

    std::map<boost::fibers::fiber::id, int> count_by_rank;

    for (const auto& rank : ranks)
    {
        count_by_rank[rank]++;
    }

    EXPECT_EQ(ranks.size(), count);
    EXPECT_EQ(count_by_rank.size(), 2);
}

TEST_F(TestPipeline, UnmatchedIngress)
{
    std::function<void(mrc::segment::Builder&)> init = [](mrc::segment::Builder& builder) {};

    auto pipe = pipeline::make_pipeline();

    pipe->make_segment("TestSegment1", segment::IngressPorts<int>({"some_port"}), init);

    auto opt1 = std::make_shared<mrc::Options>();
    opt1->topology().user_cpuset("0");
    opt1->topology().restrict_gpus(true);

    mrc::Executor exec1{opt1};

    EXPECT_ANY_THROW(exec1.register_pipeline(std::move(pipe)));
}

TEST_F(TestPipeline, UnmatchedEgress)
{
    std::function<void(mrc::segment::Builder&)> init = [](mrc::segment::Builder& builder) {};

    auto pipe = pipeline::make_pipeline();

    pipe->make_segment("TestSegment1", segment::EgressPorts<int>({"some_port"}), init);

    auto opt1 = std::make_shared<mrc::Options>();
    opt1->topology().user_cpuset("0");
    opt1->topology().restrict_gpus(true);

    mrc::Executor exec1{opt1};

    EXPECT_ANY_THROW(exec1.register_pipeline(std::move(pipe)));
}

TEST_F(TestPipeline, RequiresMoreManifolds)
{
    std::function<void(mrc::segment::Builder&)> init = [](mrc::segment::Builder& builder) {};

    auto pipe = pipeline::make_pipeline();

    pipe->make_segment("TestSegment1", segment::EgressPorts<int>({"some_port"}), init);
    pipe->make_segment("TestSegment2", segment::IngressPorts<int>({"some_port"}), init);
    pipe->make_segment("TestSegment3", segment::IngressPorts<int>({"some_port"}), init);

    auto opt1 = std::make_shared<mrc::Options>();
    opt1->topology().user_cpuset("0");
    opt1->topology().restrict_gpus(true);

    mrc::Executor exec1{opt1};

    EXPECT_ANY_THROW(exec1.register_pipeline(std::move(pipe)));
}

class Buffer
{
  public:
    Buffer() = default;

    DELETE_COPYABILITY(Buffer);

    std::size_t* data()
    {
        return m_buffer.data();
    }

    const std::size_t* data() const
    {
        return m_buffer.data();
    }

    std::size_t size() const
    {
        return m_buffer.size();
    }

  private:
    std::array<std::size_t, 1024> m_buffer;
};

TEST_F(TestPipeline, ReusablePool)
{
    auto pool = data::ReusablePool<Buffer>::create(32);

    EXPECT_EQ(pool->size(), 0);

    for (int i = 0; i < 10; i++)
    {
        pool->add_item(std::make_unique<Buffer>());
    }

    EXPECT_EQ(pool->size(), 10);

    auto item = pool->await_item();

    EXPECT_EQ(pool->size(), 10);

    item->data()[0] = 42.0;
}

TEST_F(TestPipeline, ReusableSource)
{
    auto pipe = pipeline::make_pipeline();
    auto pool = data::ReusablePool<Buffer>::create(32);

    auto opt = std::make_shared<mrc::Options>();
    opt->topology().user_cpuset("0");
    opt->topology().restrict_gpus(true);

    mrc::Executor exec{opt};

    EXPECT_EQ(pool->size(), 0);

    for (int i = 0; i < 10; i++)
    {
        pool->add_item(std::make_unique<Buffer>());
    }

    auto init = [&exec, pool](segment::Builder& segment) {
        auto src =
            segment.make_source<data::Reusable<Buffer>>("src", [pool](rxcpp::subscriber<data::Reusable<Buffer>> s) {
                while (s.is_subscribed())
                {
                    auto buffer = pool->await_item();
                    s.on_next(std::move(buffer));
                }
                s.on_completed();
            });

        auto sink =
            segment.make_sink<data::SharedReusable<Buffer>>("sink", [&exec](data::SharedReusable<Buffer> buffer) {
                static std::size_t counter = 0;
                if (counter++ > 100)
                {
                    exec.stop();
                }
            });

        EXPECT_TRUE(src->is_runnable());
        EXPECT_TRUE(sink->is_runnable());

        segment.make_edge(src, sink);
    };

    pipe->make_segment("TestSegment1", init);
    exec.register_pipeline(std::move(pipe));

    exec.start();
    exec.join();
}

TEST_F(TestPipeline, Nodes1k)
{
    auto pipeline = pipeline::make_pipeline();

    std::size_t count      = 1000;
    std::size_t recv_count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    auto end   = std::chrono::high_resolution_clock::now();

    auto segment = pipeline->make_segment("seg_1", [&](segment::Builder& s) {
        auto rx_source = s.make_source<int>("rx_source", [&start, count](rxcpp::subscriber<int> s) {
            VLOG(1) << runnable::Context::get_runtime_context().info();
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < count; i++)
            {
                s.on_next(0);
            }
            s.on_completed();
        });
        auto queue     = s.make_object("queue", std::make_unique<node::Queue<int>>());
        s.make_edge(rx_source, queue);
        auto node = s.make_node<int>("node_0", rxcpp::operators::map([](int data) { return (data + 1); }));
        s.make_edge(queue, node);
        for (int i = 1; i < 997; i++)
        {
            std::stringstream ss;
            ss << "node_" << i;
            auto curr = s.make_node<int>(ss.str(), rxcpp::operators::map([](int data) { return (data + 1); }));
            s.make_edge(node, curr);
            node = curr;
        }
        auto rx_sink = s.make_sink<int>(
            "rx_sink",
            [&](int data) {
                VLOG(1) << "data: " << data;
                EXPECT_EQ(data, 997);
                recv_count++;
            },
            [&] {
                end = std::chrono::high_resolution_clock::now();
                EXPECT_EQ(recv_count, count);
            });
        s.make_edge(node, rx_sink);
    });

    run_manager(std::move(pipeline));

    LOG(INFO) << " time in us: " << std::chrono::duration<double>(end - start).count();
}

TEST_F(TestPipeline, EngineFactories)
{
    auto topology = mrc::internal::system::Topology::Create();

    if (topology->core_count() < 8)
    {
        GTEST_SKIP_("Not enough logical CPUs to run the test. 8 required");
    }

    auto options = std::make_shared<Options>();
    options->placement().resources_strategy(PlacementResources::Dedicated);
    options->engine_factories().set_ignore_hyper_threads(true);
    options->engine_factories().set_engine_factory_options("rivermax_threads", [](EngineFactoryOptions& opts) {
        opts.engine_type   = mrc::runnable::EngineType::Thread;
        opts.cpu_count     = 4;
        opts.allow_overlap = false;
        opts.reusable      = false;
    });

    options->engine_factories().set_engine_factory_options("stage_1", [](EngineFactoryOptions& opts) {
        opts.engine_type   = mrc::runnable::EngineType::Fiber;
        opts.cpu_count     = 2;
        opts.allow_overlap = false;
        opts.reusable      = false;
    });

    options->engine_factories().set_engine_factory_options("stage_2", [](EngineFactoryOptions& opts) {
        opts.engine_type   = mrc::runnable::EngineType::Fiber;
        opts.cpu_count     = 2;
        opts.allow_overlap = false;
        opts.reusable      = false;
    });

    Executor executor(options);
    executor.register_pipeline(test::pipelines::finite_single_segment());
    executor.start();
    executor.join();
}
