/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_mrc.hpp"

#include "mrc/channel/status.hpp"  // for Status
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/executor.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/types.hpp"  // for EngineType
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/system/device_info.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <string>
#include <thread>
#include <utility>

// IWYU thinks we need vector for make_segment
// IWYU pragma: no_include <vector>

using namespace std::chrono_literals;

namespace mrc {

TEST_CLASS(Node);

struct ParallelTests : public testing::TestWithParam<int>
{};

// Run parallel tests for 1, 2 and 4 threads
INSTANTIATE_TEST_SUITE_P(TestNode, ParallelTests, testing::Values(1, 2, 4));

TEST_F(TestNode, GenericEndToEnd)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        DVLOG(1) << "In Initializer" << std::endl;

        auto sourceStr1 = seg.make_source<std::string>("src1", [&](rxcpp::subscriber<std::string>& s) {
            std::string my_str = "One1";
            s.on_next(std::move(my_str));
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

        auto intermediate = seg.make_node<std::string, int>("intermediate",
                                                            rxcpp::operators::map([](const std::string& x) -> int {
                                                                return x.size();
                                                            }));

        seg.make_edge(sourceStr2, intermediate);

        auto sinkInt = seg.make_sink<int>(
            "sinkInt",
            [&](const int& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(intermediate, sinkInt);

        // Create 2 upstream sources and check that on_completed is called after
        // all sources have been exhausted
        auto sinkStr = seg.make_sink<std::string>(
            "sink",
            [&](const std::string& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                ++next_count;
            },
            [&](std::exception_ptr x) {
                DVLOG(1) << "Sink on_error" << std::endl;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(sourceStr1, sinkStr);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 6);
    EXPECT_EQ(complete_count, 2);
}

TEST_F(TestNode, GenericEndToEndComponent)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        DVLOG(1) << "In Initializer" << std::endl;

        auto sourceStr1 = seg.make_source<std::string>("src1", [&](rxcpp::subscriber<std::string>& s) {
            std::string my_str = "One1";
            s.on_next(std::move(my_str));
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

        auto intermediate = seg.make_node<std::string, int>("intermediate",
                                                            rxcpp::operators::map([](const std::string& x) -> int {
                                                                return x.size();
                                                            }));

        seg.make_edge(sourceStr2, intermediate);

        auto sinkInt = seg.make_sink<int>(
            "sinkInt",
            [&](const int& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(intermediate, sinkInt);

        // Create 2 upstream sources and check that on_completed is called after
        // all sources have been exhausted
        auto sinkStr = seg.make_sink_component<std::string>(
            "sink",
            [&](const std::string& x) {
                // Get the context. Should be equal to the intermediate context
                auto& ctx = runnable::Context::get_runtime_context();

                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                ++next_count;
            },
            [&](std::exception_ptr x) {
                DVLOG(1) << "Sink on_error" << std::endl;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(sourceStr1, sinkStr);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 6);
    EXPECT_EQ(complete_count, 2);
}

// ======= Replace SourceRoundRobinPolicy with approprate Operator =======
// TEST_F(TestNode, EnsureMoveSemantics)
// {
//     auto p = mrc::make_pipeline();

//     std::atomic<int> next_count     = 0;
//     std::atomic<int> complete_count = 0;

//     CopyMoveCounter::reset();

//     auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
//         auto source = seg.make_source<CopyMoveCounter>("src1", [&](rxcpp::subscriber<CopyMoveCounter>& s) {
//             s.on_next(CopyMoveCounter(1));
//             s.on_next(CopyMoveCounter(2));
//             s.on_next(CopyMoveCounter(3));
//             s.on_next(CopyMoveCounter(4));
//             s.on_completed();
//         });

//         auto intermediate = seg.make_node<CopyMoveCounter, CopyMoveCounter>(
//             "intermediate", rxcpp::operators::map<CopyMoveCounter, CopyMoveCounter>([](CopyMoveCounter&& x) {
//                 x.inc();
//                 return std::move(x);
//             }));

//         // Set writer policy to avoid duplicates
//         intermediate->get_segment_source()->set_writer_policy(
//             std::make_shared<SourceRoundRobinPolicy<CopyMoveCounter>>());

//         seg.make_edge(source, intermediate);

//         auto sinkRef = seg.make_sink<CopyMoveCounter>(
//             "sinkRef",
//             [&](CopyMoveCounter&& x) {
//                 // Print value
//                 DVLOG(1) << "Sink got value: '" << x.value() << "'" << std::endl;
//                 EXPECT_NE(x.value(), -1);
//                 ++next_count;
//             },
//             [&]() {
//                 ++complete_count;
//                 DVLOG(1) << "Sink on_completed" << std::endl;
//             });

//         auto sinkMove = seg.make_sink<CopyMoveCounter>(
//             "sinkMove",
//             [&](const CopyMoveCounter& x) {
//                 // Print value
//                 DVLOG(1) << "Sink got value: '" << x.value() << "'" << std::endl;
//                 EXPECT_NE(x.value(), -1);
//                 ++next_count;
//             },
//             [&]() {
//                 ++complete_count;
//                 DVLOG(1) << "Sink on_completed" << std::endl;
//             });

//         seg.make_edge(intermediate, sinkRef);
//         seg.make_edge(intermediate, sinkMove);
//     });

//     auto options = std::make_unique<Options>();
//     options->topology().user_cpuset("0");

//     Executor exec(std::move(options));

//     exec.register_pipeline(std::move(p));

//     exec.start();

//     exec.join();

//     DVLOG(1) << "Default count: '" << CopyMoveCounter::global_default_constructed_count << "'" << std::endl;
//     DVLOG(1) << "Create count: '" << CopyMoveCounter::global_value_constructed_count << "'" << std::endl;
//     DVLOG(1) << "Move count: '" << CopyMoveCounter::global_move_count() << "'" << std::endl;
//     DVLOG(1) << "Copy count: '" << CopyMoveCounter::global_copy_count() << "'" << std::endl;

//     EXPECT_EQ(CopyMoveCounter::global_value_constructed_count, 4);
//     EXPECT_EQ(CopyMoveCounter::global_copy_count(), 0);
// }

TEST_F(TestNode, SourceEpilogue)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;
    std::atomic<int> tap_count      = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_next(4);
            s.on_completed();
        });

        source->object().add_epilogue_tap([&tap_count](const int& x) {
            // Increment the tap count
            ++tap_count;
        });

        auto sink = seg.make_sink<int>(
            "sinkRef",
            [&](const int& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                EXPECT_NE(x, -1);
                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(source, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 4);
    EXPECT_EQ(complete_count, 1);
    EXPECT_EQ(tap_count, 4);
}

TEST_F(TestNode, SinkPrologue)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;
    std::atomic<int> tap_count      = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_next(4);
            s.on_completed();
        });

        auto sink = seg.make_sink<int>(
            "sinkRef",
            [&](const int& x) {
                // Print value
                DVLOG(1) << "Sink got value: '" << x << "'" << std::endl;
                EXPECT_NE(x, -1);
                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        sink->object().add_prologue_tap([&tap_count](const int& x) {
            // Increment the tap count
            ++tap_count;
        });

        seg.make_edge(source, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(next_count, 4);
    EXPECT_EQ(complete_count, 1);
    EXPECT_EQ(tap_count, 4);
}

TEST_F(TestNode, NodePrologueEpilogue)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> sink_sum         = 0;
    std::atomic<int> complete_count   = 0;
    std::atomic<int> prologue_tap_sum = 0;
    std::atomic<int> epilogue_tap_sum = 0;

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_next(4);
            s.on_completed();
        });

        auto node = seg.make_node<int>("node", rxcpp::operators::map([](const int& x) {
                                           // Double the value
                                           return x * 2;
                                       }));

        node->object().add_prologue_tap([&prologue_tap_sum](const int& x) {
            // Increment the tap count
            prologue_tap_sum += x;
        });

        node->object().add_epilogue_tap([&epilogue_tap_sum](const int& x) {
            // Increment the tap count
            epilogue_tap_sum += x;
        });

        seg.make_edge(source, node);

        auto sink = seg.make_sink<int>(
            "sinkRef",
            [&](const int& x) {
                // Print value
                EXPECT_NE(x, -1);
                sink_sum += x;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(node, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(sink_sum, 20);
    EXPECT_EQ(complete_count, 1);
    EXPECT_EQ(prologue_tap_sum, 10);
    EXPECT_EQ(epilogue_tap_sum, 20);
}

TEST_F(TestNode, RxNodeComponentThrows)
{
    auto p                           = mrc::make_pipeline();
    std::atomic<int> throw_count     = 0;
    std::atomic<int> sink_call_count = 0;
    std::atomic<int> complete_count  = 0;

    auto my_segment = p->make_segment("test_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("source", [&](rxcpp::subscriber<int>& s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_completed();
        });

        auto node_comp = seg.make_node_component<int, int>("node", rxcpp::operators::map([&](int i) -> int {
                                                               ++throw_count;
                                                               throw std::runtime_error("test");
                                                               return 0;
                                                           }));

        auto sink = seg.make_sink<int>(
            "sinkInt",
            [&](const int& x) {
                ++sink_call_count;
            },
            [&]() {
                ++complete_count;
            });

        seg.make_edge(source, node_comp);
        seg.make_edge(node_comp, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset("0");

    Executor exec(std::move(options));
    exec.register_pipeline(std::move(p));
    exec.start();

    EXPECT_THROW(exec.join(), std::runtime_error);

    EXPECT_EQ(throw_count, 1);
    EXPECT_EQ(sink_call_count, 0);
    EXPECT_EQ(complete_count, 0);
}

// the parallel tests:
// - SourceMultiThread
// - SinkMultiThread
// - NodeMultiThread
//
// require at least N logical cpus assigned to partition 0 where N = GetParam()
// these tests will fail if the paritioning strategy the machine on which these
// tests are running do not have sufficient logcial cpus. updated the options to
// relax partitioning to get the largest logical cpu host pools per device partition

TEST_P(ParallelTests, SourceMultiThread)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    std::mutex mut;

    std::set<std::thread::id> thread_ids;

    size_t source_count        = 10;
    size_t source_thread_count = GetParam();

    ParallelTester parallel_test(source_thread_count);

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            auto& context = mrc::runnable::Context::get_runtime_context();

            for (size_t i = 0; i < source_count; ++i)
            {
                {
                    std::lock_guard<std::mutex> lock(mut);

                    thread_ids.insert(std::this_thread::get_id());
                }

                DVLOG(1) << context.info() << " Enqueueing value: '" << i << "'" << std::endl;
                ASSERT_TRUE(parallel_test.wait_for(250ms));

                s.on_next(i);
            }

            s.on_completed();
        });

        source->launch_options().pe_count = source_thread_count;

        auto sink = seg.make_sink<int>(
            "sinkRef",
            [&](const int& x) {
                auto& context = mrc::runnable::Context::get_runtime_context();

                // Print value
                DVLOG(1) << context.info() << " Sink got value: '" << x << "'" << std::endl;

                // std::this_thread::sleep_for(500ms);

                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        // sink->launch_options().threads = 1;

        // sink->object().add_prologue_tap([&tap_count](const int& x) {
        //     // Increment the tap count
        //     ++tap_count;
        // });

        seg.make_edge(source, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset(MRC_CONCAT_STR("0-" << source_thread_count));
    options->topology().restrict_gpus(true);
    options->placement().resources_strategy(PlacementResources::Shared);  // ignore numa

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    // EXPECT_EQ(next_count, 4);
    // EXPECT_EQ(complete_count, 1);
    // EXPECT_EQ(tap_count, 4);

    EXPECT_EQ(thread_ids.size(), source_thread_count);
    EXPECT_EQ(next_count, source_thread_count * source_count);
}

TEST_P(ParallelTests, SinkMultiThread)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    std::mutex mut;

    std::set<std::thread::id> thread_ids;

    size_t thread_count = GetParam();
    size_t source_count = 10 * thread_count;

    ParallelTester parallel_test(thread_count);

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            auto& context = mrc::runnable::Context::get_runtime_context();

            for (size_t i = 0; i < source_count; ++i)
            {
                DVLOG(1) << context.info() << " Enqueueing value: '" << i << "'" << std::endl;
                s.on_next(i);
            }

            s.on_completed();
        });

        auto sink = seg.make_sink<int>(
            "sink",
            [&](const int& x) {
                auto& context = mrc::runnable::Context::get_runtime_context();

                {
                    std::lock_guard<std::mutex> lock(mut);

                    thread_ids.insert(std::this_thread::get_id());
                }

                // Print value
                DVLOG(1) << context.info() << " Sink got value: '" << x << "'" << std::endl;
                EXPECT_TRUE(parallel_test.wait_for(250ms));

                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        sink->launch_options().pe_count = thread_count;

        seg.make_edge(source, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset(MRC_CONCAT_STR("0-" << thread_count));
    options->topology().restrict_gpus(true);
    options->placement().resources_strategy(PlacementResources::Shared);  // ignore numa

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(thread_ids.size(), thread_count);
    EXPECT_EQ(next_count, source_count);
    EXPECT_EQ(complete_count, thread_count);
}

TEST_P(ParallelTests, NodeMultiThread)
{
    auto p = mrc::make_pipeline();

    std::atomic<int> next_count     = 0;
    std::atomic<int> complete_count = 0;

    std::mutex mut;

    std::set<std::thread::id> thread_ids;

    size_t thread_count = GetParam();
    size_t source_count = 10 * thread_count;

    ParallelTester parallel_test(thread_count);

    auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
        auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
            auto& context = mrc::runnable::Context::get_runtime_context();

            for (size_t i = 0; i < source_count; ++i)
            {
                DVLOG(1) << context.info() << " Enqueueing value: '" << i << "'" << std::endl;
                s.on_next(i);
            }

            s.on_completed();
        });

        auto node = seg.make_node<int>("node", rxcpp::operators::map([&](const int& x) {
                                           auto& context = mrc::runnable::Context::get_runtime_context();

                                           {
                                               std::lock_guard<std::mutex> lock(mut);

                                               thread_ids.insert(std::this_thread::get_id());
                                           }

                                           DVLOG(1) << context.info() << " Node got value: '" << x << "'" << std::endl;
                                           boost::this_fiber::yield();
                                           EXPECT_TRUE(parallel_test.wait_for(250ms));
                                           // Double the value
                                           return x * 2;
                                       }));

        node->launch_options().pe_count = thread_count;

        seg.make_edge(source, node);

        auto sink = seg.make_sink<int>(
            "sink",
            [&](const int& x) {
                auto& context = mrc::runnable::Context::get_runtime_context();

                DVLOG(1) << context.info() << " Sink got value: '" << x << "'" << std::endl;

                ++next_count;
            },
            [&]() {
                ++complete_count;
                DVLOG(1) << "Sink on_completed" << std::endl;
            });

        seg.make_edge(node, sink);
    });

    auto options = std::make_unique<Options>();
    options->topology().user_cpuset(MRC_CONCAT_STR("0-" << thread_count));
    options->topology().restrict_gpus(true);
    options->placement().resources_strategy(PlacementResources::Shared);  // ignore numa

    Executor exec(std::move(options));

    exec.register_pipeline(std::move(p));

    exec.start();

    exec.join();

    EXPECT_EQ(thread_ids.size(), thread_count);
    EXPECT_EQ(next_count, source_count);
    EXPECT_EQ(complete_count, 1);
}

struct PeExceedsResourcesParams
{
    std::string cpu_set;
    std::size_t pe_count;
    std::string bad_node;
    mrc::runnable::EngineType engine_type;
};

struct PeExceedsTests : public testing::TestWithParam<PeExceedsResourcesParams>
{};

// Run parallel tests for 1, 2 and 4 threads
INSTANTIATE_TEST_SUITE_P(TestNode,
                         PeExceedsTests,
                         testing::Values(PeExceedsResourcesParams{"0-9",  // cpu set of 10 cores
                                                                  11,     // pe count one greater
                                                                  "sink",
                                                                  mrc::runnable::EngineType::Fiber},
                                         PeExceedsResourcesParams{"0-5",  // cpu set of 6 cores
                                                                  11,
                                                                  "source",
                                                                  mrc::runnable::EngineType::Fiber},
                                         PeExceedsResourcesParams{"0-3",  // cpu set of 4 cores
                                                                  5,      // pe lower than cores
                                                                  "node",
                                                                  mrc::runnable::EngineType::Fiber},
                                         PeExceedsResourcesParams{"0-9",  // cpu set of 10 cores
                                                                  11,     // pe count one greater
                                                                  "sink",
                                                                  mrc::runnable::EngineType::Thread},
                                         PeExceedsResourcesParams{"0-5",  // cpu set of 6 cores
                                                                  11,
                                                                  "source",
                                                                  mrc::runnable::EngineType::Thread},
                                         PeExceedsResourcesParams{"0-3",  // cpu set of 4 cores
                                                                  5,      // pe lower than cores
                                                                  "node",
                                                                  mrc::runnable::EngineType::Thread}));

TEST_P(PeExceedsTests, PeExceedsResources)
{
    const auto test_params = GetParam();

    EXPECT_DEATH_OR_THROW(
        {
            // EXPECT_DEATH Executes this test in a fork, we need to reset the NvmlState singleton which would
            // otherwise reflect the state of the parent process
            mrc::system::DeviceInfo::Reset();
            auto p = mrc::make_pipeline();

            const std::string cpu_set  = test_params.cpu_set;
            const std::size_t pe_count = test_params.pe_count;

            auto my_segment = p->make_segment("my_segment", [&](segment::IBuilder& seg) {
                auto source = seg.make_source<int>("src1", [&](rxcpp::subscriber<int>& s) {
                    EXPECT_TRUE(false) << "Preflight checks should fail, this should not be "
                                          "called";
                });

                auto node = seg.make_node<int>("node", rxcpp::operators::map([&](const int& x) {
                                                   EXPECT_TRUE(false) << "Preflight checks should fail, this should "
                                                                         "not be "
                                                                         "called";
                                                   return x;
                                               }));

                auto sink = seg.make_sink<int>(
                    "sink",
                    [&](const int& x) {
                        EXPECT_TRUE(false) << "Preflight checks should fail, this should not be "
                                              "called";
                    },
                    [&]() {
                        EXPECT_TRUE(false) << "Preflight checks should fail, this should not be "
                                              "called";
                    });

                if (test_params.bad_node == "source")
                {
                    source->launch_options().pe_count = pe_count;
                }
                else if (test_params.bad_node == "node")
                {
                    node->launch_options().pe_count = pe_count;
                }
                else
                {
                    sink->launch_options().pe_count = pe_count;
                }

                seg.make_edge(source, node);
                seg.make_edge(node, sink);
            });

            auto options = std::make_unique<Options>();
            options->topology().user_cpuset(cpu_set);
            options->engine_factories().set_default_engine_type(test_params.engine_type);

            Executor exec(std::move(options));
            exec.register_pipeline(std::move(p));

            exec.start();
            exec.join();
        },
        "A node was destructed which still had dependent connections|Aborted",
        std::runtime_error);
}

}  // namespace mrc
