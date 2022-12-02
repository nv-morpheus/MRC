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

#include "common_pipelines.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/definition.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/object.hpp"

#include <boost/hana/if.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

using namespace mrc;

namespace test::pipelines {

std::unique_ptr<pipeline::Pipeline> finite_multisegment()
{
    auto pipeline = pipeline::make_pipeline();

    auto segment_initializer = [](segment::Builder& seg) {};

    // ideally we make this a true source (seg_1) and true source (seg_4)
    auto seg_1 = segment::Definition::create("seg_1", segment::EgressPorts<int>({"my_int2"}), [](segment::Builder& s) {
        auto src    = s.make_source<int>("rx_source", [](rxcpp::subscriber<int> s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_completed();
        });
        auto egress = s.get_egress<int>("my_int2");
        s.make_edge(src, egress);
    });
    auto seg_2 = segment::Definition::create("seg_2",
                                             segment::IngressPorts<int>({"my_int2"}),
                                             segment::EgressPorts<int>({"my_int3"}),
                                             [](segment::Builder& s) {
                                                 // pure pass-thru
                                                 auto in  = s.get_ingress<int>("my_int2");
                                                 auto out = s.get_egress<int>("my_int3");
                                                 s.make_edge(in, out);
                                             });
    auto seg_3 = segment::Definition::create("seg_3",
                                             segment::IngressPorts<int>({"my_int3"}),
                                             segment::EgressPorts<int>({"my_int4"}),
                                             [](segment::Builder& s) {
                                                 // pure pass-thru
                                                 auto in  = s.get_ingress<int>("my_int3");
                                                 auto out = s.get_egress<int>("my_int4");
                                                 s.make_edge(in, out);
                                             });
    auto seg_4 = segment::Definition::create("seg_4", segment::IngressPorts<int>({"my_int4"}), [](segment::Builder& s) {
        // pure pass-thru
        auto in   = s.get_ingress<int>("my_int4");
        auto sink = s.make_sink<float>("rx_sink", rxcpp::make_observer_dynamic<int>([&](int x) { LOG(INFO) << x; }));
        s.make_edge(in, sink);
    });

    pipeline->register_segment(seg_1);
    pipeline->register_segment(seg_2);
    pipeline->register_segment(seg_3);
    pipeline->register_segment(seg_4);

    return pipeline;
}

}  // namespace test::pipelines
