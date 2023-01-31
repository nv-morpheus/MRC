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

#include "test_modules.hpp"

#include "mrc/core/executor.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/modules/mirror_tap/mirror_tap_util.hpp"
#include "mrc/modules/module_registry.hpp"
#include "mrc/modules/stream_buffer/stream_buffer_immediate.hpp"
#include "mrc/modules/stream_buffer/stream_buffer_module.hpp"
#include "mrc/options/options.hpp"
#include "mrc/segment/builder.hpp"

#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>

#include <random>
#include <utility>
#include <vector>

using namespace mrc;

using StreamBufferModuleImmediate =
    modules::StreamBufferModule<std::string, modules::stream_buffers::StreamBufferImmediate>;  // NOLINT

TEST_F(TestStreamBufferModule, InitailizationTest)
{
    using namespace modules;

    auto init_wrapper = [](segment::Builder& builder) {
        auto config1        = nlohmann::json();
        auto mirror_buffer1 = builder.make_module<StreamBufferModuleImmediate>("mirror_tap", config1);
    };

    m_pipeline->make_segment("Initialization_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.stop();
    executor.join();
}

TEST_F(TestStreamBufferModule, SinglePipelineImmediateStreamBufferRawThroughputTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineImmediateStreamBufferRawThroughputTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{100000};
    unsigned int packets_main{0};

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::Builder& builder) {
        auto source = builder.make_source<std::string>(test_name + "_main_source",
                                                       [packet_count](rxcpp::subscriber<std::string>& sub) {
                                                           if (sub.is_subscribed())
                                                           {
                                                               for (unsigned int i = 0; i < packet_count; i++)
                                                               {
                                                                   sub.on_next(std::to_string(packet_count));
                                                               }
                                                           }

                                                           sub.on_completed();
                                                       });

        auto config = nlohmann::json{{"buffer_size", 1024}};

        auto stream_buffer = builder.make_module<StreamBufferModuleImmediate>("stream_buffer", config);
        builder.make_edge(source, stream_buffer->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });
        builder.make_edge(stream_buffer->output_port("output"), sink);
    };

    m_pipeline->make_segment("Main_Segment", init_wrapper_main);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    VLOG(1) << "Dropped packets: " << packet_count - packets_main << " -> "
            << (packet_count - packets_main) / (double)packet_count * 100.0 << "%";

    EXPECT_GE(packet_count, packets_main * 0.5);
}

TEST_F(TestStreamBufferModule, SinglePipelineImmediateStreamBufferConstantRateThroughputTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineImmediateStreamBufferConstantRateThroughputTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{10000};
    unsigned int packets_main{0};

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::Builder& builder) {
        auto source = builder.make_source<std::string>(
            test_name + "_main_source",
            [packet_count](rxcpp::subscriber<std::string>& sub) {
                if (sub.is_subscribed())
                {
                    for (unsigned int i = 0; i < packet_count; i++)
                    {
                        sub.on_next(std::to_string(packet_count));
                        boost::this_fiber::sleep_for(std::chrono::nanoseconds(100));
                    }
                }

                sub.on_completed();
            });

        auto config = nlohmann::json{{"buffer_size", 1024}};

        auto stream_buffer = builder.make_module<StreamBufferModuleImmediate>("stream_buffer", config);
        builder.make_edge(source, stream_buffer->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });
        builder.make_edge(stream_buffer->output_port("output"), sink);
    };

    m_pipeline->make_segment("Main_Segment", init_wrapper_main);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    VLOG(1) << "Dropped packets: " << packet_count - packets_main << " -> "
            << (packet_count - packets_main) / (double)packet_count * 100.0 << "%";

    EXPECT_GE(packet_count, packets_main * 0.5);
}

TEST_F(TestStreamBufferModule, SinglePipelineImmediateStreamBufferVariableRateThroughputTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineImmediateStreamBufferBurstThroughputTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{100000};
    unsigned int packets_main{0};

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::Builder& builder) {
        auto source = builder.make_source<std::string>(
            test_name + "_main_source",
            [packet_count](rxcpp::subscriber<std::string>& sub) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> sleep_dis_ms(10, 200);

                if (sub.is_subscribed())
                {
                    for (unsigned int i = 0; i < packet_count; i++)
                    {
                        sub.on_next(std::to_string(packet_count));
                        std::size_t sleep_ms = sleep_dis_ms(gen);
                        boost::this_fiber::sleep_for(std::chrono::nanoseconds(sleep_ms));
                    }
                }

                sub.on_completed();
            });

        auto config = nlohmann::json{{"buffer_size", 1024}};

        auto stream_buffer = builder.make_module<StreamBufferModuleImmediate>("stream_buffer", config);
        builder.make_edge(source, stream_buffer->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });
        builder.make_edge(stream_buffer->output_port("output"), sink);
    };

    m_pipeline->make_segment("Main_Segment", init_wrapper_main);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    VLOG(1) << "Dropped packets: " << packet_count - packets_main << " -> "
            << (packet_count - packets_main) / (double)packet_count * 100.0 << "%";

    EXPECT_GE(packet_count, packets_main * 0.5);
}

TEST_F(TestStreamBufferModule, SinglePipelineImmediateStreamBufferBurstThroughputTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineImmediateStreamBufferBurstThroughputTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{100000};
    unsigned int packets_main{0};

    auto config = nlohmann::json{{"buffer_size", 1024}};

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::Builder& builder) {
        auto source = builder.make_source<std::string>(
            test_name + "_main_source",
            [packet_count](rxcpp::subscriber<std::string>& sub) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> count_dis(0, 10000);
                std::uniform_int_distribution<> sleep_dis_ms(0, 10);

                if (sub.is_subscribed())
                {
                    std::size_t emitted = 0;
                    while (emitted < packet_count)
                    {
                        std::size_t burst_count = count_dis(gen);
                        for (unsigned int i = 0; i < burst_count && emitted < packet_count; i++)
                        {
                            sub.on_next(std::to_string(packet_count));
                            emitted++;
                        }

                        std::size_t sleep_ms = sleep_dis_ms(gen);
                        boost::this_fiber::sleep_for(std::chrono::milliseconds(sleep_ms));
                    }
                }

                sub.on_completed();
            });

        auto config = nlohmann::json{{"buffer_size", 1024}};

        auto stream_buffer = builder.make_module<StreamBufferModuleImmediate>("stream_buffer", config);
        builder.make_edge(source, stream_buffer->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });
        builder.make_edge(stream_buffer->output_port("output"), sink);
    };

    m_pipeline->make_segment("Main_Segment", init_wrapper_main);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    VLOG(1) << "Dropped packets: " << packet_count - packets_main << " -> "
            << (packet_count - packets_main) / (double)packet_count * 100.0 << "%";

    EXPECT_GE(packet_count, packets_main * 0.5);
}
