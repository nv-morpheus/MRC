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

#include "test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/channel/status.hpp"
#include "mrc/core/executor.hpp"
#include "mrc/core/runtime.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/context.hpp"
#include "mrc/segment/definition.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/segment.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

// IWYU thinks we need exception & vector for segment.make_source
// IWYU pragma: no_include <exception>
// IWYU pragma: no_include <vector>

// IWYU thinks we need <algorithm>, <itertor>, <map> & <set> for initializer lists
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <iterator>
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <set>

class TestExecutor : public ::testing::Test
{
  protected:
    void SetUp() override {}

    void TearDown() override {}

    // static std::pair<std::unique_ptr<pipeline::Pipeline>, std::shared_ptr<TestSource<std::string>>>
    // make_two_node_pipeline()
    // {
    //     auto pipeline = pipeline::make_pipeline();
    //     auto source   = std::make_shared<TestSource<std::string>>("source", "channel");
    //     auto sink     = std::make_shared<TestSink<std::string>>("sink", "channel");

    //     pipeline->register_segments(source, sink);

    //     return std::make_pair(std::move(pipeline), std::move(source));
    // }

    static std::unique_ptr<pipeline::Pipeline> make_pipeline()
    {
        auto pipeline = pipeline::make_pipeline();

        auto segment_initializer = [](segment::Builder& seg) {};

        // ideally we make this a true source (seg_1) and true source (seg_4)
        auto seg_1 = Segment::create("seg_1", segment::EgressPorts<int>({"my_int2"}), [](segment::Builder& s) {
            auto src    = s.make_source<int>("rx_source", [](rxcpp::subscriber<int> s) {
                s.on_next(1);
                s.on_next(2);
                s.on_next(3);
                s.on_completed();
            });
            auto egress = s.get_egress<int>("my_int2");
            s.make_edge(src, egress);
        });
        auto seg_2 = Segment::create("seg_2",
                                     segment::IngressPorts<int>({"my_int2"}),
                                     segment::EgressPorts<int>({"my_int3"}),
                                     [](segment::Builder& s) {
                                         // pure pass-thru
                                         auto in  = s.get_ingress<int>("my_int2");
                                         auto out = s.get_egress<int>("my_int3");
                                         s.make_edge(in, out);
                                     });
        auto seg_3 = Segment::create("seg_3",
                                     segment::IngressPorts<int>({"my_int3"}),
                                     segment::EgressPorts<int>({"my_int4"}),
                                     [](segment::Builder& s) {
                                         // pure pass-thru
                                         auto in  = s.get_ingress<int>("my_int3");
                                         auto out = s.get_egress<int>("my_int4");
                                         s.make_edge(in, out);
                                     });
        auto seg_4 = Segment::create("seg_4", segment::IngressPorts<int>({"my_int4"}), [](segment::Builder& s) {
            // pure pass-thru
            auto in   = s.get_ingress<int>("my_int4");
            auto sink = s.make_sink<float>("rx_sink", rxcpp::make_observer_dynamic<int>([&](int x) {
                                               // Write to the log
                                               LOG(INFO) << x;
                                           }));
            s.make_edge(in, sink);
        });

        pipeline->register_segment(seg_1);
        pipeline->register_segment(seg_2);
        pipeline->register_segment(seg_3);
        pipeline->register_segment(seg_4);

        return pipeline;
    }

    static std::unique_ptr<Options> make_options()
    {
        auto options = std::make_unique<Options>();
        options->topology().user_cpuset("0-3");
        options->topology().restrict_gpus(true);
        return options;
    }
};

TEST_F(TestExecutor, LifeCycleSingleSegment)
{
    auto pipeline = pipeline::make_pipeline();

    auto options = make_options();
    options->engine_factories().set_engine_factory_options("single_use_threads", [](EngineFactoryOptions& options) {
        options.engine_type = runnable::EngineType::Thread;
        options.cpu_count   = 1;
        options.reusable    = false;
    });

    Executor executor(std::move(options));

    std::atomic<int> next_count = 0;
    std::atomic<int> src_count  = 0;
    std::atomic<int> node_count = 0;

    auto segment = Segment::create("seg_1", [&next_count, &src_count, &node_count](segment::Builder& s) {
        auto rx_source = s.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
            s.on_next(1.0f);
            s.on_next(2.0f);
            s.on_next(3.0f);
            s.on_completed();
        });

        // add epilogue count to track the number of floats emitted
        rx_source->object().add_epilogue_tap([&src_count](const float& value) { src_count++; });
        s.add_throughput_counter(rx_source);

        // we will run the source on a dedicated thread
        rx_source->launch_options().engine_factory_name = "single_use_threads";

        // add scalar node
        auto rx_node =
            s.make_node<float>("scalar_x2", rxcpp::operators::map([](float value) -> float { return 2.0 * value; }));

        rx_node->object().add_epilogue_tap([&node_count](const float& value) { node_count += 2; });
        s.add_throughput_counter(rx_node, [](const float& value) { return std::int64_t(value); });

        auto rx_sink = s.make_sink<float>("rx_sink", rxcpp::make_observer_dynamic<float>([&](float x) {
                                              LOG(INFO) << x;
                                              ++next_count;
                                          }));

        s.make_edge(rx_source, rx_node);
        s.make_edge(rx_node, rx_sink);
    });

    pipeline->register_segment(segment);

    executor.register_pipeline(std::move(pipeline));

    executor.start();
    executor.join();

    EXPECT_EQ(next_count, 3);
    EXPECT_EQ(src_count, 3);
    EXPECT_EQ(node_count, 6);

    // todo(ryan) - move to internal where we have access to the metrics registry
    // auto throughput_report = executor.runtime().metrics_registry().collect_throughput_counters();

    // EXPECT_EQ(throughput_report.size(), 2);

    // std::map<std::string, std::size_t> sorted_report;
    // for (auto& counter : throughput_report)
    // {
    //     DVLOG(10) << counter.name << ": " << counter.count;
    //     sorted_report[counter.name] = counter.count;
    // }

    // EXPECT_EQ(sorted_report.at("seg_1/rx_source"), 3);
    // EXPECT_EQ(sorted_report.at("seg_1/scalar_x2"), 12);
}

TEST_F(TestExecutor, LifeCycleSingleSegmentOpMuxer)
{
    Executor executor(make_options());

    auto pipeline = pipeline::make_pipeline();

    std::atomic<int> next_count = 0;

    auto segment = Segment::create("seg_1", [&next_count](segment::Builder& s) {
        auto rx_source = s.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
            DVLOG(1) << runnable::Context::get_runtime_context().info();
            s.on_next(1.0f);
            s.on_next(2.0f);
            s.on_next(3.0f);
            s.on_completed();
        });

        auto rx_sink = s.make_sink<float>("rx_sink", rxcpp::make_observer_dynamic<float>([&](float x) {
                                              DVLOG(1) << runnable::Context::get_runtime_context().info();
                                              LOG(INFO) << x;
                                              ++next_count;
                                          }));

        s.make_edge(rx_source, rx_sink);
    });

    pipeline->register_segment(segment);

    executor.register_pipeline(std::move(pipeline));

    executor.start();
    executor.join();

    EXPECT_EQ(next_count, 3);
}

TEST_F(TestExecutor, LifeCycleSingleSegmentOpMuxerOnThreads)
{
    auto options = make_options();
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(std::move(options));

    auto pipeline = pipeline::make_pipeline();

    std::atomic<int> next_count = 0;

    auto segment = Segment::create("seg_1", [&next_count](segment::Builder& s) {
        auto rx_source = s.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
            DVLOG(1) << runnable::Context::get_runtime_context().info();
            s.on_next(1.0f);
            s.on_next(2.0f);
            s.on_next(3.0f);
            s.on_completed();
        });

        auto rx_sink = s.make_sink<float>("rx_sink", rxcpp::make_observer_dynamic<float>([&](float x) {
                                              DVLOG(1) << runnable::Context::get_runtime_context().info();
                                              LOG(INFO) << x;
                                              ++next_count;
                                          }));

        s.make_edge(rx_source, rx_sink);
    });

    pipeline->register_segment(segment);

    executor.register_pipeline(std::move(pipeline));

    executor.start();
    executor.join();

    EXPECT_EQ(next_count, 3);
}

TEST_F(TestExecutor, LifeCycleSingleSegmentConcurrentSource)
{
    auto options = make_options();
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(std::move(options));

    auto pipeline = pipeline::make_pipeline();

    std::mutex mutex;
    std::set<std::size_t> unique_thread_ids;

    auto add_thread_id = [&](const std::size_t& id) {
        std::lock_guard<decltype(mutex)> lock(mutex);
        unique_thread_ids.insert(id);
    };

    auto segment = Segment::create("seg_1", [&add_thread_id](segment::Builder& s) {
        auto rx_source = s.make_source<std::size_t>("rx_source", [](rxcpp::subscriber<std::size_t> s) {
            auto thread_id_hash = std::hash<std::thread::id>()(std::this_thread::get_id());
            DVLOG(1) << runnable::Context::get_runtime_context().info() << ": hash=" << thread_id_hash;
            s.on_next(thread_id_hash);
            s.on_completed();
        });

        rx_source->launch_options().pe_count       = 2;
        rx_source->launch_options().engines_per_pe = 2;

        auto rx_sink =
            s.make_sink<std::size_t>("rx_sink", rxcpp::make_observer_dynamic<std::size_t>([&](std::size_t x) {
                                         DVLOG(1) << runnable::Context::get_runtime_context().info();
                                         LOG(INFO) << x;
                                         add_thread_id(x);
                                     }));

        s.make_edge(rx_source, rx_sink);
    });

    pipeline->register_segment(segment);

    executor.register_pipeline(std::move(pipeline));

    executor.start();
    executor.join();

    EXPECT_EQ(unique_thread_ids.size(), 4);
}

TEST_F(TestExecutor, LifeCycleSingleSegmentConcurrentSourceWithStaggeredShutdown)
{
    auto options = make_options();
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(std::move(options));

    auto pipeline = pipeline::make_pipeline();

    std::mutex mutex;
    std::set<std::size_t> unique_thread_ids;

    auto add_thread_id = [&](const std::size_t& id) {
        std::lock_guard<decltype(mutex)> lock(mutex);
        unique_thread_ids.insert(id);
    };

    auto segment = Segment::create("seg_1", [&add_thread_id](segment::Builder& s) {
        auto rx_source = s.make_source<std::size_t>("rx_source", [](rxcpp::subscriber<std::size_t> s) {
            auto thread_id_hash = std::hash<std::thread::id>()(std::this_thread::get_id());
            auto& ctx           = runnable::Context::get_runtime_context();
            DVLOG(1) << ctx.info() << ": hash=" << thread_id_hash;
            if (ctx.rank() > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(ctx.rank() * 100));
            }
            EXPECT_TRUE(s.is_subscribed());
            s.on_next(thread_id_hash);
            s.on_completed();
        });

        rx_source->launch_options().pe_count       = 2;
        rx_source->launch_options().engines_per_pe = 2;

        auto rx_sink =
            s.make_sink<std::size_t>("rx_sink", rxcpp::make_observer_dynamic<std::size_t>([&](std::size_t x) {
                                         DVLOG(1) << runnable::Context::get_runtime_context().info();
                                         LOG(INFO) << x;
                                         add_thread_id(x);
                                     }));

        s.make_edge(rx_source, rx_sink);
    });

    pipeline->register_segment(segment);

    executor.register_pipeline(std::move(pipeline));

    executor.start();
    executor.join();

    EXPECT_EQ(unique_thread_ids.size(), 4);
}

TEST_F(TestExecutor, LifeCycle)
{
    GTEST_SKIP() << "#185";

    Executor executor(make_options());
    executor.register_pipeline(make_pipeline());

    executor.start();
    executor.stop();
    executor.join();
}

TEST_F(TestExecutor, LifeCycleArchitect)
{
    GTEST_SKIP();

    auto options = make_options();
    options->architect_url("127.0.0.1:13337");
    options->enable_server(true);

    Executor executor(std::move(options));
    executor.register_pipeline(make_pipeline());

    executor.start();
    executor.stop();
    executor.join();
}

TEST_F(TestExecutor, MultiNode)
{
    GTEST_SKIP();

    auto options_1 = make_options();
    auto options_2 = make_options();

    options_1->architect_url("127.0.0.1:13337");
    options_1->enable_server(true);
    options_1->config_request("seg_1,seg_4");

    options_2->architect_url("127.0.0.1:13337");
    options_2->topology().user_cpuset("1");
    options_2->config_request("seg_2,seg_3");

    Executor machine_1(std::move(options_1));
    Executor machine_2(std::move(options_2));

    auto pipeline_1 = make_pipeline();
    auto pipeline_2 = make_pipeline();

    machine_1.register_pipeline(std::move(pipeline_1));
    machine_2.register_pipeline(std::move(pipeline_2));

    auto start_1 = boost::fibers::async([&] { machine_1.start(); });
    auto start_2 = boost::fibers::async([&] { machine_2.start(); });

    start_1.get();
    start_2.get();

    // the only thing that matter in this scenario is that machine_1 is the last to join
    // since it owns the oracle server

    // todo(ryan) - write tests that have all machines stop together, or have them stop in
    // discrete groups (similar to below) - the test below is more rigorous in that it test
    // 3 updates: a start, an incomplete update where 1 machines goes away while the other
    // is paused, and a final shutdown.
    // really, we should add a test that removes 1 machine while the other stays up, then
    // creates a new machine to replace the one that went away and see a resumption of pipeline
    // functionality, then do the shutdown.

    machine_2.stop();
    machine_1.stop();

    machine_2.join();
    machine_1.join();
}

// TEST_F(TestExecutor, MultiNodeTwoSegmentExample)
// {
//     GTEST_SKIP();

//     auto options_1 = make_options();
//     auto options_2 = make_options();

//     options_1->architect_url("127.0.0.1:13337");
//     options_1->enable_server(true);
//     options_1->config_request("source");

//     options_2->architect_url("127.0.0.1:13337");
//     options_2->topology().user_cpuset("1");
//     options_2->config_request("sink");

//     Executor machine_1(std::move(options_1));
//     Executor machine_2(std::move(options_2));

//     auto [pipeline_1, source_1] = make_two_node_pipeline();
//     auto [pipeline_2, source_2] = make_two_node_pipeline();

//     machine_1.register_pipeline(std::move(pipeline_1));
//     machine_2.register_pipeline(std::move(pipeline_2));

//     auto start_1 = boost::fibers::async([&] { machine_1.start(); });
//     auto start_2 = boost::fibers::async([&] { machine_2.start(); });

//     start_1.get();
//     start_2.get();

//     // the only thing that matter in this scenario is that machine_1 is the last to join
//     // since it owns the oracle server

//     source_1->channel().push("hello");
//     source_1->channel().push("mrc");
//     source_1->channel().close();

//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     // todo(ryan) - write tests that have all machines stop together, or have them stop in
//     // discrete groups (similar to below) - the test below is more rigorous in that it test
//     // 3 updates: a start, an incomplete update where 1 machines goes away while the other
//     // is paused, and a final shutdown.
//     // really, we should add a test that removes 1 machine while the other stays up, then
//     // creates a new machine to replace the one that went away and see a resumption of pipeline
//     // functionality, then do the shutdown.

//     machine_2.stop();
//     machine_2.join();

//     machine_1.stop();

//     machine_1.join();
// }

// TEST_F(TestExecutor, MultiNodeTwoSegmentNaturalShutdown)
// {
//     GTEST_SKIP();

//     auto options_1 = make_options();
//     auto options_2 = make_options();

//     options_1->architect_url("127.0.0.1:13337");
//     options_1->enable_server(true);
//     options_1->config_request("source");

//     options_2->architect_url("127.0.0.1:13337");
//     options_2->topology().user_cpuset("1");
//     options_2->config_request("sink");

//     Executor machine_1(std::move(options_1));
//     Executor machine_2(std::move(options_2));

//     auto [pipeline_1, source_1] = make_two_node_pipeline();
//     auto [pipeline_2, source_2] = make_two_node_pipeline();

//     machine_1.register_pipeline(std::move(pipeline_1));
//     machine_2.register_pipeline(std::move(pipeline_2));

//     auto start_1 = boost::fibers::async([&] { machine_1.start(); });
//     auto start_2 = boost::fibers::async([&] { machine_2.start(); });

//     start_1.get();
//     start_2.get();

//     source_1->channel().push("hello");
//     source_1->channel().push("mrc");
//     source_1->channel().close();

//     machine_2.join();
//     machine_1.join();
// }

TEST_F(TestExecutor, ConfigRegex) {}
