/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_segment.hpp"

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/dynamic_batcher.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/executor.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/pipeline/segment.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/segment/ports.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std::literals::string_literals;

namespace mrc {

TEST_F(TestSegment, CreateSegmentDefinition)
{
    auto segdef = Segment::create("segment_test", m_initializer);
}

TEST_F(TestSegment, InitializeSegmentFromDefinition)
{
    auto segdef = Segment::create("segment_test", m_initializer);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

// --- //

TEST_F(TestSegment, CreateSegmentDefinitionIngressOnly)
{
    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_initializer);
}

TEST_F(TestSegment, InitializeSegmentIngressOnlyFromDefinition)
{
    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_initializer);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

// --- //

TEST_F(TestSegment, CreateSegmentDefinitionEgressOnly)
{
    auto segdef = Segment::create("segment_test", m_egress_multi_port, m_initializer);
}

TEST_F(TestSegment, InitializeSegmentEgressOnlyFromDefinition)
{
    auto segdef = Segment::create("segment_test", m_egress_multi_port, m_initializer);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

// --- //

TEST_F(TestSegment, CreateSegmentDefinitionIngressEgress)
{
    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, m_initializer);
}

TEST_F(TestSegment, InitializeSegmentIngressEgressFromDefinition)
{
    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, m_initializer);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);

    EXPECT_EQ(segdef->name(), "segment_test");
    /*
    EXPECT_EQ(seg->name(), "segment_test");
    EXPECT_EQ(seg->id(), 30063);
    EXPECT_EQ(seg->rank(), 15);
    EXPECT_EQ(seg->is_running(), false);
    */
}

TEST_F(TestSegment, PortsConstructorBadNameBuilderSizeMismatch)
{
    using port_type_t = segment::Ports<segment::IngressPortBase>;

    std::vector<std::string> port_names{"a", "b", "c"};
    std::vector<port_type_t::port_builder_fn_t> port_builder_fns{};

    EXPECT_THROW(port_type_t BadPorts(port_names, port_builder_fns), exceptions::MrcRuntimeError);
}

TEST_F(TestSegment, PortsConstructorBadDuplicateName)
{
    using port_type_t = segment::Ports<segment::IngressPortBase>;

    auto port_builder = [](const SegmentAddress& address,
                           const PortName& name) -> std::shared_ptr<segment::IngressPortBase> {
        return std::make_shared<segment::IngressPort<int>>(address, name);
    };

    std::vector<std::string> port_names{"a", "b", "a"};
    std::vector<port_type_t::port_builder_fn_t> port_builder_fns{port_builder, port_builder, port_builder};

    EXPECT_THROW(port_type_t BadPorts(port_names, port_builder_fns), exceptions::MrcRuntimeError);
}

TEST_F(TestSegment, UserLambdaIsCalled)
{
    GTEST_SKIP() << "Skipping until issue #59 is resolved";

    EXPECT_EQ(m_initializer_called, false);

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, m_initializer);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);

    EXPECT_EQ(m_initializer_called, true);
}

TEST_F(TestSegment, SegmentRxSinkCreation)
{
    auto init = [](segment::IBuilder& segment) {
        auto x = segment.make_sink<std::string>(
            "x_sink",
            [](std::string x) {
                DVLOG(1) << x << std::endl;
            },
            []() {
                DVLOG(1) << "Completed" << std::endl;
            });
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentRxSourceCreation)
{
    auto init = [](segment::IBuilder& segment) {
        auto x = segment.make_source<std::string>("x_src", [&](rxcpp::subscriber<std::string> s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        });
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentRxNodeCreation)
{
    auto init = [](segment::IBuilder& segment) {
        auto x = segment.make_node<std::string, std::string>("x");

        auto y = segment.make_node<std::string, double>("y", rxcpp::operators::map([](std::string s) -> double {
                                                            return 1.0;
                                                        }));

        auto z = segment.make_node<double, double>("z", rxcpp::operators::map([](double d) -> double {
                                                       return 2.0 * d;
                                                   }));

        auto w = segment.make_node<double, double>("w", rxcpp::operators::map([](double d) -> double {
                                                       return 2.0 * d;
                                                   }));
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentRxNodeStaticEdges)
{
    auto init = [this](segment::IBuilder& segment) {
        auto x = segment.make_node<std::string, std::string>("x");
        // Compiler error, can't create a node with no operations that has disperate input/output types
        // auto x2 = segment.make_node<std::string, double>("x");

        auto y = segment.make_node<std::string, double>("y", rxcpp::operators::map([](std::string s) -> double {
                                                            return 1.0;
                                                        }));

        auto z = segment.make_node<double, double>("z", rxcpp::operators::map([](double d) -> double {
                                                       return 2.0 * d;
                                                   }));

        auto w = segment.make_node<double, double>("w", rxcpp::operators::map([](double d) -> double {
                                                       return 2.0 * d;
                                                   }));

        segment.make_edge(x, y);
        segment.make_edge(y, z);
        segment.make_edge(z, w);
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentRxNodeValidTypeConversionWorks)
{
    auto init = [this](segment::IBuilder& segment) {
        auto x = segment.make_node<std::string, std::string>("x");
        // Compiler error, can't create a node with no operations that has disperate input/output types
        // auto x2 = segment.make_node<std::string, double>("x");

        auto y = segment.make_node<std::string, double>("y", rxcpp::operators::map([](std::string s) -> double {
                                                            return 1.0;
                                                        }));

        auto z = segment.make_node<float, float>("z", rxcpp::operators::map([](float f) -> float {
                                                     return 2.0 * f;
                                                 }));

        auto w = segment.make_node<double, double>("w", rxcpp::operators::map([](double d) -> double {
                                                       return 2.0 * d;
                                                   }));

        auto k = segment.make_node<int, int>("k", rxcpp::operators::map([](int i) -> int {
                                                 return 2.0 * i;
                                             }));

        segment.make_edge(x, y);
        segment.make_edge(y, z);
        segment.make_edge(z, w);
        segment.make_edge(w, k);
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

/* dynamic edges are python only
TEST_F(SegmentTests, SegmentRxNodeDynamicEdges)
{
    auto init = [this](segment::IBuilder& segment) {
        auto x = segment.make_node<std::string, std::string>("x");
        // Compiler error, can't create a node with no operations that has disperate input/output types
        // auto x2 = segment.make_node<std::string, double>("x");

        auto y = segment.make_node<std::string, double>(
            "y", rxcpp::operators::map([](std::string s) -> double { return 1.0; }));

        auto z =
            segment.make_node<double, double>("z", rxcpp::operators::map([](double d) -> double { return 2.0 * d; }));

        auto w =
            segment.make_node<double, double>("w", rxcpp::operators::map([](double d) -> double { return 2.0 * d; }));

        segment.make_edge("x", "y");
        segment.make_edge("y", "z");
        segment.make_edge("z", "w");

        // 6 Ingress, 4 Egress, 4 internal
        EXPECT_EQ(segment.node_count(), m_InterfaceNodeCount + 4);
    };

    auto segdef = Segment::create("segment_test", m_ingress_multi_port, m_egress_multi_port, init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}


*/

TEST_F(TestSegment, SegmentEndToEndTest)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < 10 && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto internal = segment.make_node<std::string, std::string>(
            "internal",
            rxcpp::operators::tap([](std::string s) {
                VLOG(10) << "Side Effect[Before]: " << s << std::endl;
            }),
            rxcpp::operators::map([](std::string s) {
                return s + "-Mapped";
            }),
            rxcpp::operators::tap([](std::string s) {
                VLOG(10) << "Side Effect[After]: " << s << std::endl;
            }));

        segment.make_edge(src, internal);

        auto sink = segment.make_sink<std::string>(
            "sink",
            [](std::string x) {
                VLOG(10) << x << std::endl;
            },
            []() {
                VLOG(10) << "Completed" << std::endl;
            });

        segment.make_edge(internal, sink);
    };

    auto segdef = Segment::create("segment_test", init);
    // move to internal tests to access the builder
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, CompileTimeConversionValuesWorkAsExpected)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < 10 && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto internal = segment.make_node<std::string, float>("internal",
                                                              rxcpp::operators::map([](std::string s) -> float {
                                                                  return 1.0;
                                                              }));

        segment.make_edge(src, internal);

        auto convert_1_double = segment.make_node<double, double>("convert_1_double",
                                                                  rxcpp::operators::map([](double d) -> double {
                                                                      VLOG(9) << "convert_1_double: " << d << std::endl;
                                                                      VLOG(9) << "convert_1_double: " << d * 1.1
                                                                              << std::endl;
                                                                      return d * 1.1;
                                                                  }));

        segment.make_edge(internal, convert_1_double);

        auto convert_2_int = segment.make_node<int, int>("convert_2_int", rxcpp::operators::map([](int i) -> int {
                                                             VLOG(9) << "convert_2_int: " << i << std::endl;
                                                             VLOG(9) << "convert_2_int(negative): " << -i << std::endl;
                                                             return i * 1.1 * -1.0;
                                                         }));

        segment.make_edge(convert_1_double, convert_2_int);

        auto convert_3_sizet = segment.make_node<std::size_t, std::string>(
            "convert_3_sizet",
            rxcpp::operators::map([](std::size_t szt) -> std::string {
                VLOG(9) << "convert_3_sizet: " << szt << std::endl;
                return std::to_string(szt);
            }));

        segment.make_edge(convert_2_int, convert_3_sizet);

        auto sink = segment.make_sink<std::string>(
            "sink",
            [](std::string x) {
                VLOG(10) << x << std::endl;
            },
            []() {
                VLOG(10) << "Completed" << std::endl;
            });

        segment.make_edge(convert_3_sizet, sink);
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, RuntimeConversionValuesWorkAsExpected)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < 10 && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto internal = segment.make_node<std::string, float>("internal",
                                                              rxcpp::operators::map([](std::string s) -> float {
                                                                  return 1.0;
                                                              }));

        segment.make_edge<std::string, std::string>("src", "internal");

        auto convert_1_double = segment.make_node<double, double>("convert_1_double",
                                                                  rxcpp::operators::map([](double d) -> double {
                                                                      VLOG(9) << "convert_1_double: " << d << std::endl;
                                                                      VLOG(9) << "convert_1_double: " << d * 1.1
                                                                              << std::endl;
                                                                      return d * 1.1;
                                                                  }));

        segment.make_edge<float, double>("internal", "convert_1_double");

        auto convert_2_int = segment.make_node<int, int>("convert_2_int", rxcpp::operators::map([](int i) -> int {
                                                             VLOG(9) << "convert_2_int: " << i << std::endl;
                                                             VLOG(9) << "convert_2_int(negative): " << -i << std::endl;
                                                             return i * 1.1 * -1.0;
                                                         }));

        // Note: this will disable narrowing and fail:
        //      segment.make_edge<double, int, false>("convert_1_double", "convert_2_int");
        segment.make_edge<double, int>("convert_1_double", "convert_2_int");

        auto convert_3_sizet = segment.make_node<std::size_t, std::string>(
            "convert_3_sizet",
            rxcpp::operators::map([](std::size_t szt) -> std::string {
                VLOG(9) << "convert_3_sizet: " << szt << std::endl;
                return std::to_string(szt);
            }));

        segment.make_edge<int, std::size_t>("convert_2_int", "convert_3_sizet");

        auto sink = segment.make_sink<std::string>(
            "sink",
            [](std::string x) {
                VLOG(10) << x << std::endl;
            },
            []() {
                VLOG(10) << "Completed" << std::endl;
            });

        segment.make_edge<std::string>("convert_3_sizet", "sink");
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentEndToEndTestRx)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string> s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        });

        auto internal = segment.make_node<std::string, std::string>(
            "internal",
            rxcpp::operators::tap([](std::string s) {
                VLOG(10) << "Side Effect[Before]: " << s << std::endl;
            }),
            rxcpp::operators::map([](std::string s) {
                return s + "-Mapped";
            }),
            rxcpp::operators::tap([](std::string s) {
                VLOG(10) << "Side Effect[After]: " << s << std::endl;
            }));

        segment.make_edge(src, internal);

        auto sink = segment.make_sink<std::string>(
            "sink",
            [](std::string x) {
                VLOG(10) << x << std::endl;
            },
            []() {
                VLOG(10) << "Completed" << std::endl;
            });

        segment.make_edge(internal, sink);
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

void execute_pipeline(std::unique_ptr<pipeline::IPipeline> pipeline)
{
    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");
    Executor exec(std::move(options));
    exec.register_pipeline(std::move(pipeline));
    exec.start();
    exec.join();
}

TEST_F(TestSegment, ChannelClose)
{
    auto p = mrc::make_pipeline();

    int next_count     = 0;
    int complete_count = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        DVLOG(1) << "In Initializer" << std::endl;

        auto sourceStr1 = seg.make_source<std::string>("src1", [&](rxcpp::subscriber<std::string>& s) {
            s.on_next("One1");
            s.on_next("Two1");
            s.on_next("Three1");
            s.on_completed();
        });

        auto sourceStr2 = seg.make_source<std::string>("src2", [&](rxcpp::subscriber<std::string>& s) {
            s.on_next("One2");
            s.on_next("Two2");
            s.on_next("Three2");
            s.on_completed();
        });

        // Create 2 upstream sources and check that on_completed is called after
        // all sources have been exhausted
        auto sinkStr = seg.make_sink<std::string>(
            "sink",
            [&](const std::string& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(sourceStr1, sinkStr);
        seg.make_edge(sourceStr2, sinkStr);
    });

    execute_pipeline(std::move(p));

    EXPECT_EQ(next_count, 6);
    EXPECT_EQ(complete_count, 1);
}

TEST_F(TestSegment, SegmentEndToEndTestSinkOutput)
{
    unsigned int iterations{10};
    std::atomic<unsigned int> sink_results{0};

    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < iterations && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto internal = segment.make_node<std::string, unsigned int>("internal",
                                                                     rxcpp::operators::map([](std::string s) {
                                                                         return static_cast<unsigned int>(s.size());
                                                                     }));

        segment.make_edge(src, internal);

        auto sink = segment.make_sink<unsigned int>("sink", [&sink_results](unsigned int x) {
            sink_results.fetch_add(x, std::memory_order_relaxed);
        });

        segment.make_edge(internal, sink);
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentSingleSourceTwoNodesException)
{
    unsigned int iterations{3};
    std::atomic<unsigned int> sink1_results{0};
    float sink2_results{0};
    std::mutex mux;

    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < iterations && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto str_length = segment.make_node<std::string, unsigned int>("str_length",
                                                                       rxcpp::operators::map([](std::string s) {
                                                                           DVLOG(1) << "str_length received: '" << s
                                                                                    << "'" << std::endl;
                                                                           return static_cast<unsigned int>(s.size());
                                                                       }));

        segment.make_edge(src, str_length);

        auto sink1 = segment.make_sink<unsigned int>("sink1", [&sink1_results](unsigned int x) {
            sink1_results.fetch_add(x, std::memory_order_relaxed);
        });

        segment.make_edge(str_length, sink1);

        auto str_half_length = segment.make_node<std::string, float>("str_half_length",
                                                                     rxcpp::operators::map([](std::string s) {
                                                                         DVLOG(1) << "str_half_length received: '" << s
                                                                                  << "'" << std::endl;
                                                                         return s.size() / 2.0F;
                                                                     }));

        EXPECT_ANY_THROW(segment.make_edge(src, str_half_length));

        auto sink2 = segment.make_sink<float>("sink2", [&](float x) {
            // C++20 adds fetch_add to atomic<float>
            const std::lock_guard<std::mutex> lock(mux);
            sink2_results += x;
        });

        segment.make_edge(str_half_length, sink2);
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
}

TEST_F(TestSegment, SegmentSingleSourceTwoNodes)
{
    unsigned int iterations{3};
    std::atomic<unsigned int> sink1_results{0};
    float sink2_results{0};
    std::mutex mux;

    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < iterations && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto bcast_src = segment.construct_object<node::Broadcast<std::string>>("broadcast");

        segment.make_edge(src, bcast_src);

        auto str_length = segment.make_node<std::string, unsigned int>("str_length",
                                                                       rxcpp::operators::map([](std::string s) {
                                                                           DVLOG(1) << "str_length received: '" << s
                                                                                    << "'" << std::endl;
                                                                           return static_cast<unsigned int>(s.size());
                                                                       }));

        segment.make_edge(bcast_src, str_length);

        auto sink1 = segment.make_sink<unsigned int>("sink1", [&sink1_results](unsigned int x) {
            sink1_results.fetch_add(x, std::memory_order_relaxed);
        });

        segment.make_edge(str_length, sink1);

        auto str_half_length = segment.make_node<std::string, float>("str_half_length",
                                                                     rxcpp::operators::map([](std::string s) {
                                                                         DVLOG(1) << "str_half_length received: '" << s
                                                                                  << "'" << std::endl;
                                                                         return s.size() / 2.0F;
                                                                     }));

        segment.make_edge(bcast_src, str_half_length);

        auto sink2 = segment.make_sink<float>("sink2", [&](float x) {
            // C++20 adds fetch_add to atomic<float>
            const std::lock_guard<std::mutex> lock(mux);
            sink2_results += x;
        });

        segment.make_edge(str_half_length, sink2);
    };

    auto segdef = Segment::create("segment_test", init);
    // // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);

    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));
    execute_pipeline(std::move(pipeline));

    EXPECT_EQ(sink1_results, 11 * iterations);
    EXPECT_EQ(sink2_results, 5.5F * iterations);
}

TEST_F(TestSegment, SegmentSingleSourceMultiNodes)
{
    constexpr unsigned int NumChildren{10};
    unsigned int iterations{3};

    std::mutex mux;
    std::array<unsigned int, NumChildren> sink_results;
    sink_results.fill(0);

    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&](rxcpp::subscriber<std::string>& s) {
            for (size_t i = 0; i < iterations && s.is_subscribed(); i++)
            {
                s.on_next("One");
                s.on_next("Two");
                s.on_next("Three");
            }

            s.on_completed();
        });

        auto bcast = segment.construct_object<node::Broadcast<std::string>>("broadcast");
        segment.make_edge(src, bcast);

        for (unsigned int i = 0; i < NumChildren; ++i)
        {
            std::string node_name{"str_length"s + std::to_string(i)};
            auto node = segment.make_node<std::string, unsigned int>(
                node_name,
                rxcpp::operators::map([i, node_name](std::string s) {
                    DVLOG(1) << node_name << " received: '" << s << "'" << std::endl;
                    return static_cast<unsigned int>(s.size() + i);
                }));

            segment.make_edge(bcast, node);

            std::string sink_name{"sink"s + std::to_string(i)};
            auto sink = segment.make_sink<unsigned int>(sink_name, [i, &sink_results, &mux](unsigned int x) {
                const std::lock_guard<std::mutex> lock(mux);
                sink_results[i] += x;
            });

            segment.make_edge(node, sink);
        }
    };

    auto segdef = Segment::create("segment_test", init);
    // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);

    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));

    execute_pipeline(std::move(pipeline));

    for (unsigned int i = 0; i < NumChildren; ++i)
    {
        EXPECT_EQ(sink_results[i], (11 + i * 3) * iterations);
    }
}

TEST_F(TestSegment, EnsureMove)
{
    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [](rxcpp::subscriber<std::string>& s) {
            if (s.is_subscribed())
            {
                std::string data{"this should be moved"};
                s.on_next(std::move(data));

                EXPECT_EQ(data, ""s);
            }
            else
            {
                FAIL() << "is_subscrived returned a false";
            }

            s.on_completed();
        });

        auto sink = segment.make_sink<std::string>("sink", [](std::string x) {
            EXPECT_EQ(x, "this should be moved"s);
        });

        segment.make_edge(src, sink);
    };

    auto segdef = Segment::create("segment_test", init);

    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));
    execute_pipeline(std::move(pipeline));
}

TEST_F(TestSegment, EnsureMoveMultiChildren)
{
    constexpr unsigned int NumChildren{10};
    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [](rxcpp::subscriber<std::string>& s) {
            if (s.is_subscribed())
            {
                std::string data{"this should be moved"};
                s.on_next(std::move(data));

                EXPECT_EQ(data, ""s);
            }
            else
            {
                FAIL() << "is_subscrived returned a false";
            }

            s.on_completed();
        });

        auto bcast_src = segment.construct_object<node::Broadcast<std::string>>("broadcast");
        segment.make_edge(src, bcast_src);

        for (unsigned int i = 0; i < NumChildren; ++i)
        {
            std::string node_name{"node_"s + std::to_string(i)};
            auto node = segment.make_node<std::string, unsigned int>(node_name,
                                                                     rxcpp::operators::map([i](std::string s) {
                                                                         EXPECT_EQ(s, "this should be moved"s);
                                                                         return static_cast<unsigned int>(s.size() + i);
                                                                     }));

            segment.make_edge(bcast_src, node);

            std::string sink_name{"sink"s + std::to_string(i)};
            auto sink = segment.make_sink<unsigned int>(sink_name, [i](unsigned int x) {
                DVLOG(1) << "Sink" << i << " received: " << x << std::endl;
                EXPECT_EQ(x, 20 + i);
            });

            segment.make_edge(node, sink);
        }
    };

    auto segdef   = Segment::create("segment_test", init);
    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));
    execute_pipeline(std::move(pipeline));
}

TEST_F(TestSegment, EnsureMoveConstructor)
{
    // First ensure CopyMoveCounter is working as expected
    {
        CopyMoveCounter tm;
        EXPECT_FALSE(tm.was_copied());
        EXPECT_FALSE(tm.was_moved());

        CopyMoveCounter tc{tm};
        EXPECT_TRUE(tm.was_copied());
        EXPECT_FALSE(tm.was_moved());
        EXPECT_EQ(tc.copy_count(), 1);
        EXPECT_EQ(tc.move_count(), 0);
    }

    {
        CopyMoveCounter tm;
        CopyMoveCounter tmm{std::move(tm)};
        EXPECT_FALSE(tm.was_copied());
        EXPECT_TRUE(tm.was_moved());
        EXPECT_EQ(tmm.copy_count(), 0);
        EXPECT_EQ(tmm.move_count(), 1);
    }

    {
        CopyMoveCounter tm;
        CopyMoveCounter tac = tm;
        EXPECT_TRUE(tm.was_copied());
        EXPECT_FALSE(tm.was_moved());
        EXPECT_EQ(tac.copy_count(), 1);
        EXPECT_EQ(tac.move_count(), 0);
    }

    {
        CopyMoveCounter tm;
        CopyMoveCounter tam = std::move(tm);
        EXPECT_FALSE(tm.was_copied());
        EXPECT_TRUE(tm.was_moved());
        EXPECT_EQ(tam.copy_count(), 0);
        EXPECT_EQ(tam.move_count(), 1);
    }

    {  // Test only one child
        auto init = [&](segment::IBuilder& segment) {
            auto src = segment.make_source<CopyMoveCounter>("src", [](rxcpp::subscriber<CopyMoveCounter>& s) {
                if (s.is_subscribed())
                {
                    CopyMoveCounter tm;
                    s.on_next(std::move(tm));

                    EXPECT_FALSE(tm.was_copied());
                    EXPECT_TRUE(tm.was_moved());
                }
                else
                {
                    FAIL() << "is_subscrived returned a false";
                }

                s.on_completed();
            });

            auto sink = segment.make_sink<CopyMoveCounter>("sink", [](CopyMoveCounter x) {
                EXPECT_EQ(x.copy_count(), 0);
                EXPECT_GT(x.move_count(), 0);
            });

            segment.make_edge(src, sink);
        };

        auto segdef = Segment::create("segment_test", init);
        // auto builder =  std::make_unique<segment::IBuilder>(segdef, 42);
    }

    {  // Test multiple children
        constexpr unsigned int NumChildren{10};
        auto init = [&](segment::IBuilder& segment) {
            auto src = segment.make_source<CopyMoveCounter>("src", [](rxcpp::subscriber<CopyMoveCounter>& s) {
                if (s.is_subscribed())
                {
                    CopyMoveCounter tm;
                    s.on_next(std::move(tm));

                    // We should have a move followed by a copy
                    EXPECT_TRUE(tm.was_moved());
                }
                else
                {
                    FAIL() << "is_subscrived returned a false";
                }

                s.on_completed();
            });

            auto bcast_src = segment.construct_object<node::Broadcast<CopyMoveCounter>>("broadcast");
            segment.make_edge(src, bcast_src);

            for (unsigned int i = 0; i < NumChildren; ++i)
            {
                std::string sink_name{"sink"s + std::to_string(i)};
                auto sink = segment.make_sink<CopyMoveCounter>(sink_name, [i](CopyMoveCounter x) {
                    // the copies will come in with copy count of 0
                    // the original will come in with a copy count of > 0
                    EXPECT_GE(x.copy_count(), 0);
                });

                segment.make_edge(bcast_src, sink);
            }
        };

        auto segdef   = Segment::create("segment_test", init);
        auto pipeline = mrc::make_pipeline();
        pipeline->register_segment(std::move(segdef));
        execute_pipeline(std::move(pipeline));
    }
}

TEST_F(TestSegment, SegmentTestRxcppHigherLevelNodes)
{
    std::size_t iterations = 5;
    using mrc::benchmarking::TraceStatistics;
    TraceStatistics::trace_channels();
    TraceStatistics::trace_operators();

    auto init = [&iterations](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("src", [&iterations](rxcpp::subscriber<std::string> s) {
            for (auto i = 0; i < iterations; ++i)
            {
                s.on_next("One_" + std::to_string(i));
            }
            s.on_completed();
        });

        auto internal_1 = segment.make_node<std::string, std::string>(
            "internal_1",
            rxcpp::operators::map([](std::string s) {
                std::vector<std::string> ret = {s + "_1", s + "_2", s + "_3"};
                VLOG(10) << "i1 - tick" << std::endl;
                return ret;
            }),
            rxcpp::operators::concat_map(
                // Unrolling observer -- required
                [](std::vector<std::string> svec) {
                    return rxcpp::observable<>::create<std::string>([svec](rxcpp::subscriber<std::string> sub) {
                        try
                        {
                            for (auto x : svec)
                            {
                                sub.on_next(x);
                            }
                            sub.on_completed();
                        } catch (...)
                        {
                            sub.on_error(rxcpp::util::current_exception());
                        }
                    });
                }),
            rxcpp::operators::map([](std::string s) {
                VLOG(10) << s << std::endl;
                return s;
            }));

        segment.make_edge(src, internal_1);

        auto internal_2 = segment.make_node<std::string, std::string>("internal_2",
                                                                      rxcpp::operators::map([](std::string s) {
                                                                          return s;
                                                                      }));

        segment.make_edge(internal_1, internal_2);

        auto sink = segment.make_sink<std::string>(
            "sink",
            [](std::string x) {
                VLOG(10) << x << std::endl;
            },
            []() {
                VLOG(10) << "Completed" << std::endl;
            });

        segment.make_edge(internal_2, sink);
    };

    TraceStatistics::reset();

    auto segdef   = Segment::create("segment_stats_test", init);
    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));
    execute_pipeline(std::move(pipeline));

    nlohmann::json j = TraceStatistics::aggregate();
    auto _j          = j["aggregations"]["components"]["metrics"];
    // std::cerr << j.dump(2);
    EXPECT_EQ(_j.contains("/segment_stats_test/src"), true);
    auto src_json = j["/segment_stats_test/src"];
    // stat_check_helper(src_json, 0, 0, iterations, iterations);

    EXPECT_EQ(_j.contains("/segment_stats_test/internal_1"), true);
    auto i1_json = j["/segment_stats_test/internal_1"];
    // stat_check_helper(i1_json, iterations, iterations, iterations, iterations);

    EXPECT_EQ(_j.contains("/segment_stats_test/internal_2"), true);
    auto i2_json = j["/segment_stats_test/internal_1"];
    // stat_check_helper(i2_json, iterations, iterations, iterations, iterations);

    EXPECT_EQ(_j.contains("/segment_stats_test/sink"), true);
    auto sink_json = j["/segment_stats_test/sink"];
    // stat_check_helper(sink_json, iterations, iterations, 0, 0);
    TraceStatistics::reset();
}

TEST_F(TestSegment, SegmentGetEgressError)
{
    auto init = [](segment::IBuilder& segment) {
        segment.get_egress<int>("test");
    };
    auto segdef = Segment::create("segment_test", init);

    /*
    try
    {
        auto seg = Segment::instantiate(*segdef, m_resources);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(e.what(), std::string("Node 'test' doesn't exist in the segment."));
    } catch (...)
    {
        FAIL() << "Expected std::invalid_argument";
    }
    */
}

TEST_F(TestSegment, SegmentGetEgressNotEgressError)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("test", [&](rxcpp::subscriber<std::string>& s) {
            s.on_completed();
        });
        segment.get_egress<int>("test");
    };
    auto segdef = Segment::create("segment_test", init);

    /*
    try
    {
        auto seg = Segment::instantiate(*segdef, m_resources);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(e.what(), std::string("Object 'test' is not an egress"));
    } catch (...)
    {
        FAIL() << "Expected std::invalid_argument";
    }
    */
}

TEST_F(TestSegment, SegmentDynamicBatcher)
{
    unsigned int iterations{3};
    std::atomic<unsigned int> sink1_results{0};
    float sink2_results{0};
    std::mutex mux;

    auto init = [&](segment::IBuilder& segment) {
        auto src = segment.make_source<int>("src", [&](rxcpp::subscriber<int>& s) {
            for (size_t i = 0; i < iterations && s.is_subscribed(); i++)
            {
                s.on_next(1);
                s.on_next(2);
                s.on_next(3);
            }

            s.on_completed();
        });

        auto dynamic_batcher = segment.construct_object<node::DynamicBatcher<int, runnable::Context>>("dynamic_batcher", 2, std::chrono::milliseconds(100));

        segment.make_edge(src, dynamic_batcher);

        auto sink = segment.make_sink<std::vector<int>>("sink", [&](std::vector<int> x) {
            DVLOG(1) << "Sink got vector" << std::endl;
            for (auto i : x)
            {
                DVLOG(1) << "Sink got value: " << i << std::endl;
                // sink1_results.fetch_add(i, std::memory_order_relaxed);
            }
        });

        segment.make_edge(dynamic_batcher, sink);
    };

    auto segdef = Segment::create("dynamic_batcher_test", init);

    auto pipeline = mrc::make_pipeline();
    pipeline->register_segment(std::move(segdef));
    execute_pipeline(std::move(pipeline));
}

}  // namespace mrc
