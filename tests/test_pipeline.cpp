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
#include "mrc/memory/memory_kind.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/definition.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/segment.hpp"

#include <gtest/gtest.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-operators.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

// IWYU thinks we need std::copy, exception, begin/end, map, set & vector for initializer lists
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <exception>
// IWYU pragma: no_include <iterator>
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <set>
// IWYU pragma: no_include <vector>

using namespace mrc;

class TestPipeline : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline = std::move(pipeline::make_pipeline());
        m_options  = std::make_unique<Options>();

        m_options->topology().user_cpuset("0");
    }

    void TearDown() override {}

    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    std::unique_ptr<Options> m_options;
};

TEST_F(TestPipeline, LifeCycle)
{
    std::atomic<std::uint32_t> counter = 0;

    // note: i'm not sure if the top-level segment definition require types or just names
    // types might be useful for both count enforcement/zipping type-to-name, but also
    // make the instantiation of the serialization/deserialization network stack simplier
    auto segment_initializer = [&counter](segment::Builder& seg) {
        ++counter;
        // Segment initialization code here.
    };

    auto segment = segment::Definition::create("seg_1",
                                               segment::IngressPorts<int>({"my_int"}),
                                               segment::EgressPorts<float>({"my_float"}),
                                               segment_initializer);

    m_pipeline->register_segment(segment);
    // EXPECT_EQ(m_pipeline->segment_count(), 1);
    EXPECT_EQ(counter.load(), 0);

    // our test executor will let us break down the an Executor::run
    // into more testable components
    // in this case, we will issue a command that should create 1 instance
    // of each segment in the pipeline
    // TestExecutor executor;
    // executor.register_pipeline(std::move(pipeline));

    // executor.create_segments();
    // EXPECT_EQ(counter.load(), 1);
    // EXPECT_EQ(executor.segment_count(), 1);
    // EXPECT_EQ(executor.registered_ingress_port_count(), 1);
}

TEST_F(TestPipeline, DuplicateSegments)
{
    auto segment_initializer = [](segment::Builder& seg) {};
    auto seg_1               = segment::Definition::create(
        "seg_1", segment::IngressPorts<int>({"my_int1"}), segment::EgressPorts<int>({"my_int2"}), segment_initializer);
    m_pipeline->register_segment(seg_1);
    EXPECT_ANY_THROW(m_pipeline->register_segment(seg_1));
}

TEST_F(TestPipeline, TwoSegment)
{
    GTEST_SKIP() << "#185";

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto pipeline = pipeline::make_pipeline();

    auto seg_1 =
        pipeline->make_segment("seg_1", segment::EgressPorts<float>({"float_port"}), [](segment::Builder& seg) {
            auto rx_source = seg.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
                LOG(INFO) << "emit 1";
                s.on_next(1.0f);
                LOG(INFO) << "emit 2";
                s.on_next(2.0f);
                LOG(INFO) << "emit 3";
                s.on_next(3.0f);
                LOG(INFO) << "issuing complete";
                s.on_completed();
            });

            auto my_float_egress = seg.get_egress<float>("float_port");

            seg.make_edge(rx_source, my_float_egress);
        });

    auto seg_2 =
        pipeline->make_segment("seg_2", segment::IngressPorts<float>({"float_port"}), [&](segment::Builder& seg) {
            auto my_float_ingress = seg.get_ingress<float>("float_port");

            auto rx_sink = seg.make_sink<float>("rx_sink",
                                                rxcpp::make_observer_dynamic<float>(
                                                    [&](float x) {
                                                        DVLOG(1) << x << std::endl;
                                                        ++next_count;
                                                    },
                                                    [&]() {
                                                        DVLOG(1) << "Completed" << std::endl;
                                                        ++complete_count;
                                                    }));

            seg.make_edge(my_float_ingress, rx_sink);
        });

    Executor exec(std::move(m_options));

    exec.register_pipeline(std::move(pipeline));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 3);
    EXPECT_EQ(complete_count, 1);

    LOG(INFO) << "Done" << std::endl;
}

/*
TEST_F(TestPipeline, TwoSegmentManualTag)
{
    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto pipeline = pipeline::make_pipeline();

    auto seg_1 =
        pipeline->make_segment("seg_1", segment::EgressPorts<Tagged<float>>({"float_port"}), [](segment::Builder& seg) {
            auto rx_source = seg.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
                s.on_next(1.0f);
                s.on_next(2.0f);
                s.on_next(3.0f);
                s.on_completed();
            });

            auto seg_2_addr = seg.make_address("seg_2");

            auto tagging_node =
                seg.make_node<float, Tagged<float>>("tagging_node", rxcpp::operators::map([seg_2_addr](float x) {
                                                           // Always specify local here
                                                           return Tagged<float>({seg_2_addr}, x);
                                                       }));

            seg.make_edge(rx_source, tagging_node);

            auto my_float_egress = seg.get_egress<Tagged<float>>("float_port");

            seg.make_edge(tagging_node, my_float_egress);
        });

    auto seg_2 =
        pipeline->make_segment("seg_2", segment::IngressPorts<float>({"float_port"}), [&](segment::Builder& seg) {
            auto my_float_ingress = seg.get_ingress<float>("float_port");

            auto rx_sink = seg.make_sink<float>("rx_sink",
                                                   rxcpp::make_observer_dynamic<float>(
                                                       [&](float x) {
                                                           DVLOG(1) << x << std::endl;
                                                           ++next_count;
                                                       },
                                                       [&]() {
                                                           DVLOG(1) << "Completed" << std::endl;
                                                           ++complete_count;
                                                       }));

            seg.make_edge(my_float_ingress, rx_sink);
        });

    auto serialize_pipeline = pipeline->serialize();
    // EXPECT_TRUE(ConfigurationManager::validate_pipeline(serialize_pipeline));

    Executor exec(std::move(m_options));

    exec.register_pipeline(std::move(pipeline));
    exec.start();
    exec.join();

    EXPECT_EQ(next_count, 3);
    EXPECT_EQ(complete_count, 1);

    LOG(INFO) << "Done" << std::endl;
}


TEST_F(TestPipeline, TwoSegmentManualTagImmediateStop)
{
    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto pipeline = pipeline::make_pipeline();

    auto seg_1 =
        pipeline->make_segment("seg_1", segment::EgressPorts<Tagged<float>>({"float_port"}), [](segment::Builder& seg) {
            auto rx_source =
                seg.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) { s.on_completed(); });

            auto seg_2_addr = seg.make_address("seg_2");

            auto tagging_node =
                seg.make_node<float, Tagged<float>>("tagging_node", rxcpp::operators::map([seg_2_addr](float x) {
                                                           // Always specify local here
                                                           return Tagged<float>({seg_2_addr}, x);
                                                       }));

            seg.make_edge(rx_source, tagging_node);

            auto my_float_egress = seg.get_egress<Tagged<float>>("float_port");

            seg.make_edge(tagging_node, my_float_egress);
        });

    auto seg_2 =
        pipeline->make_segment("seg_2", segment::IngressPorts<float>({"float_port"}), [&](segment::Builder& seg) {
            auto my_float_ingress = seg.get_ingress<float>("float_port");

            auto rx_sink = seg.make_sink<float>("rx_sink",
                                                   rxcpp::make_observer_dynamic<float>(
                                                       [&](float x) {
                                                           DVLOG(1) << x << std::endl;
                                                           ++next_count;
                                                       },
                                                       [&]() {
                                                           DVLOG(1) << "Completed" << std::endl;
                                                           ++complete_count;
                                                       }));

            seg.make_edge(my_float_ingress, rx_sink);
        });

    auto serialize_pipeline = pipeline->serialize();
    // EXPECT_TRUE(ConfigurationManager::validate_pipeline(serialize_pipeline));

    Executor exec(std::move(m_options));

    exec.register_pipeline(std::move(pipeline));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 0);
    EXPECT_EQ(complete_count, 1);

    LOG(INFO) << "Done" << std::endl;
}
*/

/*
TEST_F(TestPipeline, Architect)
{
    // GTEST_SKIP();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto pipeline = pipeline::make_pipeline();

    auto seg_1 = pipeline->make_segment("seg_1", EgressPorts<float>({"float_port"}), [](segment::Builder& seg) {
        auto rx_source = seg.make_source<float>("rx_source", [](rxcpp::subscriber<float> s) {
            s.on_next(1.0f);
            s.on_next(2.0f);
            s.on_next(3.0f);
            s.on_completed();
        });

        auto my_float_egress = seg.get_egress<float>("float_port");

        seg.make_edge(rx_source, my_float_egress);
    });

    auto seg_2 = pipeline->make_segment("seg_2", IngressPorts<float>({"float_port"}), [&](segment::Builder& seg) {
        auto my_float_ingress = seg.get_ingress<float>("float_port");

        auto rx_sink = seg.make_sink<float>("rx_sink",
                                               rxcpp::make_observer_dynamic<float>(
                                                   [&](float x) {
                                                       DVLOG(1) << x << std::endl;
                                                       ++next_count;
                                                   },
                                                   [&]() {
                                                       DVLOG(1) << "Completed" << std::endl;
                                                       ++complete_count;
                                                   }));

        seg.make_edge(my_float_ingress, rx_sink);
    });

    ArchitectExecutor exec(std::move(m_options));

    exec.register_pipeline(std::move(pipeline));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 3);
    EXPECT_EQ(complete_count, 1);

    LOG(INFO) << "Done" << std::endl;
}
*/
