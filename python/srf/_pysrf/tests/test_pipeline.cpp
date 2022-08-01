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

#include "test_pysrf.hpp"

#include "pysrf/executor.hpp"  // IWYU pragma: keep
#include "pysrf/pipeline.hpp"
#include "pysrf/port_builders.hpp"
#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/channel/status.hpp"
#include "srf/core/executor.hpp"
#include "srf/core/utils.hpp"
#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/manifold/egress.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/options/options.hpp"
#include "srf/options/topology.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"

#include <boost/hana/if.hpp>
#include <ext/alloc_traits.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include <boost/fiber/future/detail/shared_state.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/task_base.hpp>
// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"
// IWYU pragma: no_include "rx-includes.hpp"

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;
using namespace py::literals;

PYSRF_TEST_CLASS(Pipeline);

// TEST_F(TestPipeline, Constructor)
// {
//     pysrf::Pipeline p;

//     // since the underlying pipeline is private swapping it
//     // out to see what the underlying pipeline was
//     auto pipe_ptr = p.swap();
//     EXPECT_NE(pipe_ptr, nullptr);

//     EXPECT_EQ(pipe_ptr->segment_count(), 0);
//     EXPECT_TRUE(pipe_ptr->sources().empty());
//     EXPECT_TRUE(pipe_ptr->sinks().empty());

//     EXPECT_THROW(pipe_ptr->lookup_id("fake"s), std::exception);
// }

// TEST_F(TestPipeline, MakeSegment)
// {
//     pysrf::Pipeline p;
//     p.make_segment("turtle"s, [](srf::segment::Builder& seg) {});
//     p.make_segment("lizard"s, [](srf::segment::Builder& seg) {});
//     p.make_segment("frog"s, [](srf::segment::Builder& seg) {});

//     auto pipe_ptr = p.swap();
//     EXPECT_EQ(pipe_ptr->segment_count(), 3);

//     auto segId = pipe_ptr->lookup_id("lizard"s);
//     EXPECT_EQ(pipe_ptr->lookup_name(segId), "lizard"s);
// }

TEST_F(TestPipeline, Execute)
{
    std::atomic<unsigned int> counter = 0;
    pysrf::Pipeline p;

    auto init = [&counter](srf::segment::Builder& seg) {
        auto src = seg.make_source<bool>("src", [](rxcpp::subscriber<bool>& s) {
            if (s.is_subscribed())
            {
                s.on_next(true);
                s.on_next(false);
            }

            s.on_completed();
        });

        auto internal = seg.make_node<bool, unsigned int>("internal", rxcpp::operators::map([](bool b) {
                                                              unsigned int i{b};
                                                              return i;
                                                          }));

        auto sink = seg.make_sink<unsigned int>(
            "sink", [&counter](unsigned int x) { counter.fetch_add(x, std::memory_order_relaxed); });

        seg.make_edge(src, internal);
        seg.make_edge(internal, sink);
    };

    p.make_segment("seg1"s, init);
    p.make_segment("seg2"s, init);
    p.make_segment("seg3"s, init);

    auto options = std::make_shared<srf::Options>();
    options->topology().user_cpuset("0");

    // note this is the base SRF executor not a pysrf executor
    srf::Executor exec{options};
    exec.register_pipeline(p.swap());

    py::gil_scoped_release release;
    exec.start();
    exec.join();

    EXPECT_EQ(counter, 3);
}

TEST_F(TestPipeline, DynamicPortConstructionGood)
{
    pysrf::PortBuilderUtil::register_port_util<pysrf::PyHolder>();

    std::string name                                 = "xyz";
    std::function<void(srf::segment::Builder&)> init = [](srf::segment::Builder& builder) {
        std::cerr << "Builder called" << std::endl;
    };

    std::vector<std::vector<std::string>> ingress_id_vec;
    std::vector<std::vector<std::string>> egress_id_vec;

    for (int i = 0; i <= 5; ++i)
    {
        std::vector<std::string> isubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "i" << i << j;
            isubvec.push_back(sstream.str());
        }
        ingress_id_vec.push_back(isubvec);
    }

    for (int i = 0; i <= 5; ++i)
    {
        std::vector<std::string> esubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "e" << i << j;
            esubvec.push_back(sstream.str());
        }
        egress_id_vec.push_back(esubvec);
    }

    for (auto ivec : ingress_id_vec)
    {
        for (auto evec : egress_id_vec)
        {
            pysrf::Pipeline pipe;
            pipe.make_segment(name, py::cast(ivec), py::cast(evec), init);
        }
    }
}

TEST_F(TestPipeline, DynamicPortConstructionBadDuplicatePorts)
{
    pysrf::PortBuilderUtil::register_port_util<pysrf::PyHolder>();

    std::string name                                 = "xyz";
    std::function<void(srf::segment::Builder&)> init = [](srf::segment::Builder& builder) {
        std::cerr << "Builder called" << std::endl;
    };

    std::vector<std::vector<std::string>> ingress_id_vec;
    std::vector<std::vector<std::string>> egress_id_vec;

    for (int i = 0; i <= 5; ++i)
    {
        std::vector<std::string> isubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "i" << i << j;
            isubvec.push_back(sstream.str());
        }
        ingress_id_vec.push_back(isubvec);
    }

    for (int i = 0; i <= 5; ++i)
    {
        std::vector<std::string> esubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "e" << i << j;
            esubvec.push_back(sstream.str());
        }
        egress_id_vec.push_back(esubvec);
    }

    pysrf::Pipeline pipe;
    EXPECT_THROW(pipe.make_segment(name, py::cast(ingress_id_vec), py::list(), init), std::runtime_error);
    EXPECT_THROW(pipe.make_segment(name, py::list(), py::cast(egress_id_vec), init), std::runtime_error);
    EXPECT_THROW(pipe.make_segment(name, py::cast(ingress_id_vec), py::cast(egress_id_vec), init), std::runtime_error);
}

TEST_F(TestPipeline, DynamicPortsIngressEgressMultiSegmentSingleExecutor)
{
    const std::size_t object_count{10};
    const std::size_t source_count{4};
    std::atomic<std::size_t> sink_count{0};
    std::vector<std::string> source_segment_egress_ids{"source_1", "source_2", "source_3", "source_4"};
    std::vector<std::string> intermediate_segment_egress_ids{"internal_1", "internal_2", "internal_3", "internal_4"};

    std::function<void(srf::segment::Builder&)> seg1_init =
        [source_segment_egress_ids](srf::segment::Builder& builder) {
            for (int i = 0; i < source_segment_egress_ids.size(); i++)
            {
                auto src = builder.make_source<pysrf::PyHolder>(
                    "stage1_source_" + std::to_string(i), [](rxcpp::subscriber<pysrf::PyHolder>& subscriber) {
                        if (subscriber.is_subscribed())
                        {
                            py::gil_scoped_acquire gil;
                            for (int i = 0; i < object_count; ++i)
                            {
                                pysrf::PyHolder object = py::dict("prop1"_a = "abc", "prop2"_a = 1, "prop3"_a = 8910);
                                {
                                    py::gil_scoped_release nogil;
                                    subscriber.on_next(std::move(object));
                                }
                            }
                        }
                        subscriber.on_completed();
                    });

                py::gil_scoped_acquire gil;
                auto egress_test = builder.get_egress<pysrf::PyHolder>(source_segment_egress_ids[i]);
                EXPECT_TRUE(source_segment_egress_ids[i] == egress_test->name());
                EXPECT_TRUE(egress_test->is_sink());
                builder.make_edge(src, egress_test);
            }

            LOG(INFO) << "Finished TestSegment1 Initialization";
        };

    std::function<void(srf::segment::Builder&)> seg2_init =
        [source_segment_egress_ids, intermediate_segment_egress_ids](srf::segment::Builder& builder) {
            for (auto ingress_it : source_segment_egress_ids)
            {
                auto ingress_test = builder.get_ingress<pysrf::PyHolder>(ingress_it);
                EXPECT_TRUE(ingress_it == ingress_test->name());
                EXPECT_TRUE(ingress_test->is_source());
            }

            for (int i = 0; i < source_segment_egress_ids.size(); ++i)
            {
                auto ingress = builder.get_ingress<pysrf::PyHolder>(source_segment_egress_ids[i]);
                auto egress  = builder.get_egress<pysrf::PyHolder>(intermediate_segment_egress_ids[i]);

                builder.make_edge(ingress, egress);
            }
            LOG(INFO) << "Finished TestSegment2 Initialization";
        };

    std::function<void(srf::segment::Builder&)> seg3_init =
        [&sink_count, intermediate_segment_egress_ids](srf::segment::Builder& builder) {
            for (int i = 0; i < intermediate_segment_egress_ids.size(); ++i)
            {
                auto ingress = builder.get_ingress<pysrf::PyHolder>(intermediate_segment_egress_ids[i]);

                auto sink = builder.make_sink<pysrf::PyHolder>("local_sink_" + std::to_string(i),
                                                               [&sink_count](pysrf::PyHolder object) {
                                                                   py::gil_scoped_acquire gil;
                                                                   sink_count++;
                                                               });

                builder.make_edge(ingress, sink);
            }
            LOG(INFO) << "Finished TestSegment3 Initialization";
        };

    pysrf::Pipeline pipe;

    pipe.make_segment("TestSegment1", py::list(), py::cast(source_segment_egress_ids), seg1_init);
    pipe.make_segment(
        "TestSegment2", py::cast(source_segment_egress_ids), py::cast(intermediate_segment_egress_ids), seg2_init);
    pipe.make_segment("TestSegment3", py::cast(intermediate_segment_egress_ids), py::list(), seg3_init);

    auto opt1 = std::make_shared<srf::Options>();
    opt1->topology().user_cpuset("0");
    opt1->topology().restrict_gpus(true);

    srf::Executor exec1{opt1};

    exec1.register_pipeline(pipe.swap());

    py::gil_scoped_release release;

    exec1.start();
    exec1.join();
    EXPECT_EQ(sink_count, source_count * object_count);
}
