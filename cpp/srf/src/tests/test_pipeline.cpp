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

#include "pipelines/common_pipelines.hpp"

#include "internal/pipeline/manager.hpp"
#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/resource_partitions.hpp"
#include "internal/system/system.hpp"
#include "internal/utils/collision_detector.hpp"

#include "nodes/common_nodes.hpp"
#include "srf/channel/forward.hpp"
#include "srf/channel/status.hpp"
#include "srf/core/addresses.hpp"
#include "srf/core/executor.hpp"
#include "srf/internal/pipeline/ipipeline.hpp"
#include "srf/internal/segment/idefinition.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/options/options.hpp"
#include "srf/options/topology.hpp"
#include "srf/pipeline/pipeline.hpp"
#include "srf/runnable/context.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/egress_ports.hpp"
#include "srf/segment/ingress_ports.hpp"
#include "srf/segment/object.hpp"
#include "srf/srf.hpp"

#include "rxcpp/rx-includes.hpp"
#include "rxcpp/rx-observer.hpp"
#include "rxcpp/rx-operators.hpp"
#include "rxcpp/rx-predef.hpp"
#include "rxcpp/rx-subscriber.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>

#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace srf;

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

    return internal::system::System::make_system(std::move(options));
}

static std::shared_ptr<internal::pipeline::Pipeline> unwrap(internal::pipeline::IPipeline& pipeline)
{
    return internal::pipeline::Pipeline::unwrap(pipeline);
}

static void run_custom_manager(std::unique_ptr<internal::pipeline::IPipeline> pipeline,
                               internal::pipeline::SegmentAddresses&& update,
                               bool delayed_stop = false)
{
    auto resources = internal::resources::make_resource_partitions(make_system([](Options& options) {
        options.topology().user_cpuset("0-1");
        options.topology().restrict_gpus(true);
    }));

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
    auto resources = internal::resources::make_resource_partitions(make_system([](Options& options) {
        options.topology().user_cpuset("0-1");
        options.topology().restrict_gpus(true);
    }));

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
    auto pipeline = srf::make_pipeline();

    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto rx_source = s.make_object("rx_source", test::nodes::infinite_int_rx_source());
        auto rx_sink   = s.make_object("rx_sink", test::nodes::int_sink_throw_on_even());
        s.make_edge(rx_source, rx_sink);
    });

    EXPECT_ANY_THROW(run_manager(std::move(pipeline)));
}

TEST_F(TestPipeline, LifeCycleStop)
{
    auto pipeline = srf::make_pipeline();

    auto segment = pipeline->make_segment("seg_1", [](segment::Builder& s) {
        auto rx_source = s.make_object("rx_source", test::nodes::infinite_int_rx_source());
        auto rx_sink   = s.make_object("rx_sink", test::nodes::int_sink());
        s.make_edge(rx_source, rx_sink);
    });

    run_manager(std::move(pipeline), true);
}

TEST_F(TestPipeline, DuplicateNameInSegment)
{
    auto pipeline = srf::make_pipeline();

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

    auto pipeline = srf::make_pipeline();

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
