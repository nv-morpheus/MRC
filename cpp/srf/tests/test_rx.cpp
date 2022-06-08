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

#include "test_srf.hpp"  // IWYU pragma: associated

#include <srf/channel/buffered_channel.hpp>
#include <srf/channel/channel.hpp>
#include <srf/channel/status.hpp>
#include <srf/pipeline/pipeline.hpp>

#include <glog/logging.h>

#include <rxcpp/operators/rx-map.hpp>  // for map
#include <rxcpp/operators/rx-tap.hpp>  // for tap
#include <rxcpp/rx-includes.hpp>       // for apply
#include <rxcpp/rx-observer.hpp>       // for is_on_next_of<>::not_void, is_on_error<>::not_void, make_observer_dynamic
#include <rxcpp/rx-predef.hpp>         // for trace_activity
#include <rxcpp/rx-subscriber.hpp>     // for make_subscriber, subscriber
#include <rxcpp/sources/rx-iterate.hpp>  // for from

#include <memory>   // seg.make_rx_node returns a shared_ptr
#include <ostream>  // for glog macros
#include <string>
#include <utility>  // for move

// IWYU thinkgs exception, uint16_t & vector are needed for seg.make_rx_source
// IWYU pragma: no_include <cstdint>
// IWYU pragma: no_include <exception>
// IWYU pragma: no_include <vector>

TEST_CLASS(Rx);

TEST_F(TestRx, WithoutSegment)
{
    // FiberGroup<> fibers(2);

    // Why is this test allowed?

    // Do we want to be able to instantiate a Segment, this means we have given it the runtime resources
    // then be able to modify the ndoes?

    /*

    Pipeline p;
    auto segdef = segment::Definition::create(p, IngressPorts<int>({"my_int"}), EgressPorts<int>({"empty"}),
    "my_segment",
                                  [](segment::Definition& seg) {});
    auto seg = Segment::instantiate(0, 0, *segdef);

    auto sourceStr = rx::RxBuilder::Source<std::string>::create(*seg, "src", [&](rxcpp::subscriber<std::string> s) {
        s.on_next("One");
        s.on_next("Two");
        s.on_next("Three");
        s.on_completed();
    });

    auto nodeStr = rx::RxBuilder::Node<std::string, std::string>::create(
        *seg, "internal",
        rxcpp::operators::tap([](std::string s) { DVLOG(1) << "Side Effect[Before]: " << s << std::endl; }),
        rxcpp::operators::map([](std::string s) { return s + "-Mapped"; }),
        rxcpp::operators::tap([](std::string s) { DVLOG(1) << "Side Effect[After]: " << s << std::endl; }));

    // ensure we have some back pressure by making the channel very small
    nodeStr.input_channel(std::move(std::make_unique<BufferedChannel<std::string>>(2)));

    make_edge(sourceStr, nodeStr);

    auto sinkStr = rx::RxBuilder::Sink<std::string>::create(
        *seg, "sink",
        rxcpp::make_observer_dynamic<std::string>([](std::string x) { DVLOG(1) << x << std::endl; },
                                                  []() { DVLOG(1) << "Completed" << std::endl; }));

    make_edge(nodeStr, sinkStr);

    sinkStr.start();
    nodeStr.start();
    sourceStr.start();

    DVLOG(1) << "Started" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    DVLOG(1) << "Stopping" << std::endl;

    sourceStr.stop();
    sourceStr.join();

    nodeStr.stop();
    nodeStr.join();

    sinkStr.stop();
    sinkStr.join();

    DVLOG(1) << "Stopped" << std::endl;

    */
}

TEST_F(TestRx, SingleSegment)
{
    auto p = pipeline::make_pipeline();

    auto my_segment = segment::Definition::create("my_segment", [](segment::Definition& seg) {
        DVLOG(1) << "In Initializer" << std::endl;

        auto sourceStr = seg.make_rx_source<std::string>("src", [&](const rxcpp::subscriber<std::string>& s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        });

        auto nodeStr = seg.make_rx_node<std::string, std::string>(
            "internal",
            rxcpp::operators::tap([](std::string s) { DVLOG(1) << "Side Effect[Before]: " << s << std::endl; }),
            rxcpp::operators::map([](std::string s) { return s + "-Mapped"; }),
            rxcpp::operators::tap([](std::string s) { DVLOG(1) << "Side Effect[After]: " << s << std::endl; }));

        // ensure we have some back pressure by making the channel very small
        nodeStr->get_segment_sink()->input_channel(std::move(std::make_unique<BufferedChannel<std::string>>(2)));

        seg.make_edge(sourceStr, nodeStr);

        auto sinkStr = seg.make_rx_sink<std::string>(
            "sink",
            rxcpp::make_observer_dynamic<std::string>([](std::string x) { DVLOG(1) << x << std::endl; },
                                                      []() { DVLOG(1) << "Completed" << std::endl; }));

        seg.make_edge(nodeStr, sinkStr);
    });

    /*

    // todo move start/stop/join into a segment tester
    // Segment::instantiate requires a resources object
    // if we want to instantiate segments in tests lower than the provider
    // for resources, then we need a more general way to providing resources

    auto seg = Segment::instantiate(*my_segment, resources);

    seg->start();

    seg->stop();

    seg->join();

    DVLOG(1) << "Stopped" << std::endl;
    */
}
