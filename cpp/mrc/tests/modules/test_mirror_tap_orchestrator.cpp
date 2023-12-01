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

#include "test_modules.hpp"

#include "mrc/cuda/device_guard.hpp"
#include "mrc/experimental/modules/mirror_tap/mirror_tap_orchestrator.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/executor.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/object.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <memory>
#include <ostream>
#include <string>
#include <utility>

using namespace mrc;

TEST_F(TestMirrorTapUtil, SinglePipelineTapAndBufferTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineTapAndBufferTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{10000};
    unsigned int packets_main{0};
    unsigned int packets_mirrored{0};

    auto config = nlohmann::json();

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::IBuilder& builder) {
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

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });

        // Untapped edge that will be broken and tapped by the mirror tap.
        builder.make_edge(source, sink);
    };

    auto init_wrapper_mirrored = [&packets_mirrored, test_name](segment::IBuilder& builder) {
        auto mirror_sink = builder.make_sink<std::string>(test_name + "_mirror_sink",
                                                          [&packets_mirrored](std::string input) {
                                                              VLOG(10) << "tick -> " << input << std::endl
                                                                       << std::flush;
                                                              packets_mirrored++;
                                                          });
    };

    auto mirror_tap = MirrorTapOrchestrator<std::string>(test_name + "mirror_tap", config);

    auto tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main,
                                                   test_name + "_main_source",
                                                   test_name + "_main_sink");

    auto tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored, test_name + "_mirror_sink");

    m_pipeline->make_segment("Main_Segment", mirror_tap.create_or_extend_egress_ports(), tapped_init_wrapper_main);

    m_pipeline->make_segment("StreamMirror_Segment",
                             mirror_tap.create_or_extend_ingress_ports(),
                             tapped_init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    // Since we wire everything up before the main source starts pumping data, we should always have the same
    // number of packets between main and mirrored, even though we're using hot observables internally.
    EXPECT_EQ(packets_main, packet_count);

    // Set this really low; it can vary wildly in CI, depending on various factors, and available threads.
    EXPECT_GE(packets_mirrored, packet_count * 0.1);
}

TEST_F(TestMirrorTapUtil, SinglePipelineTapAndBufferWithAdditionalPortsTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineTapAndBufferTestWithAdditionalPortsTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{10000};
    unsigned int packets_main{0};
    unsigned int packets_mirrored{0};
    unsigned int packets_non_mirrored{0};

    auto config = nlohmann::json();

    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::IBuilder& builder) {
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

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });
        // Untapped edge that will be broken and tapped by the mirror tap.
        builder.make_edge(source, sink);

        auto extra_source = builder.make_source<std::string>(test_name + "_main_extra_source",
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
        auto extra_egress = builder.get_egress<std::string>("non_mirror_port");
        builder.make_edge(extra_source, extra_egress);
    };

    auto init_wrapper_mirrored = [&packets_mirrored, &packets_non_mirrored, test_name](segment::IBuilder& builder) {
        auto mirror_sink     = builder.make_sink<std::string>(test_name + "_mirror_sink",
                                                          [&packets_mirrored](std::string input) {
                                                              packets_mirrored++;
                                                          });
        auto non_mirror_sink = builder.make_sink<std::string>(test_name + "_non_mirror_sink",
                                                              [&packets_non_mirrored](std::string input) {
                                                                  packets_non_mirrored++;
                                                              });
        auto extra_ingress   = builder.get_ingress<std::string>("non_mirror_port");
        builder.make_edge(extra_ingress, non_mirror_sink);
    };

    auto mirror_tap = MirrorTapOrchestrator<std::string>(test_name + "mirror_tap", config);

    auto tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main,
                                                   test_name + "_main_source",
                                                   test_name + "_main_sink");

    auto tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored, test_name + "_mirror_sink");

    auto egress_ports = segment::EgressPorts<std::string>({"non_mirror_port"});
    m_pipeline->make_segment("Main_Segment",
                             mirror_tap.create_or_extend_egress_ports(egress_ports),
                             tapped_init_wrapper_main);

    auto ingress_ports = segment::IngressPorts<std::string>({"non_mirror_port"});
    m_pipeline->make_segment("StreamMirror_Segment",
                             mirror_tap.create_or_extend_ingress_ports(ingress_ports),
                             tapped_init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    // Since we wire everything up before the main source starts pumping data, we should always have the same
    // number of packets between main and mirrored, even though we're using hot observables internally.
    EXPECT_EQ(packets_main, packet_count);
    EXPECT_EQ(packets_non_mirrored, packet_count);
    EXPECT_GE(packets_mirrored, packet_count * 0.1);
}
