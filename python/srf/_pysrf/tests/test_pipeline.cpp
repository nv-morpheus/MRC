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

#include <pysrf/pipeline.hpp>

#include <srf/channel/status.hpp>
#include <srf/core/executor.hpp>
#include <srf/node/rx_node.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/rx_source.hpp>
#include <srf/options/options.hpp>
#include <srf/options/topology.hpp>
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>

#include <gtest/gtest.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-operators.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <atomic>
#include <memory>
#include <string>

// IWYU thinks we need move & vector for auto internal = seg.make_rx_node
// IWYU pragma: no_include <utility>
// IWYU pragma: no_include <vector>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;

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
    std::string name                                 = "xyz";
    std::function<void(srf::segment::Builder&)> init = [](srf::segment::Builder& builder) {
        std::cerr << "Builder called" << std::endl;
    };

    std::vector<std::vector<std::string>> ingress_id_vec;
    std::vector<std::vector<std::string>> egress_id_vec;

    for (int i = 0; i <= SRF_MAX_INGRESS_PORTS; ++i)
    {
        std::vector<std::string> isubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "i" << i << j;
            isubvec.push_back(sstream.str());
        }
    }

    for (int i = 0; i <= SRF_MAX_EGRESS_PORTS; ++i)
    {
        std::vector<std::string> esubvec;
        for (int j = 0; j <= i; ++j)
        {
            std::stringstream sstream;
            sstream << "i" << i << j;
            esubvec.push_back(sstream.str());
        }
    }

    for (auto ivec : ingress_id_vec)
    {
        for (auto evec : egress_id_vec)
        {
            pysrf::Pipeline pipe;
            pipe.make_segment(name, ivec, evec, init);
        }
    }
}

/*
TEST_F(TestPipeline, DynamicPortsBuildIngressEgress)
{
    std::vector<std::string> ingress_port_ids{"a", "b", "c", "d"};
    std::vector<std::string> egress_port_ids{"w", "x", "y", "z"};

    std::function<void(srf::segment::Builder&)> seg1_init = [ingress_port_ids, egress_port_ids](srf::segment::Builder& builder) {
        for (auto ingress_it : ingress_port_ids) {
            auto egress_test = builder.get_egress<py::object>("test321");
            EXPECT_TRUE(std::string("test321") == egress_test->name());
            EXPECT_TRUE(egress_test->is_sink());
        }
    };

    pysrf::Pipeline seg1_pipe;

    seg1_pipe.make_segment("TestSegment1", ingress_port_ids, egress_port_ids, seg1_init);

    auto opt1 = std::make_shared<srf::Options>();
    opt1->topology().user_cpuset("0");

    srf::Executor exec1{opt1};

    exec1.register_pipeline(seg1_pipe.swap());

    py::gil_scoped_release release;
    exec1.start();
    exec1.stop();
    exec1.join();
}*/

/*
TEST_F(TestPipeline, DynamicPortsGetEgressGood)
{
    std::vector<std::string> ingress_port_ids{};
    std::vector<std::string> egress_port_ids{"test321"};

    std::function<void(srf::segment::Builder&)> seg1_init = [](srf::segment::Builder& builder) {
        auto src = builder.make_source<py::object>("src", [](rxcpp::subscriber<py::object>& s) {
            if (s.is_subscribed())
            {
                for (int i = 0; i < 3; ++i)
                {
                    py::gil_scoped_acquire acquire;
                    py::int_ data{1};
                    py::print("Created data object");
                    {
                        py::gil_scoped_release nogil;
                        s.on_next(std::move(data));
                    }
                }
            }

            s.on_completed();
        });

      auto egress_test = builder.get_egress<py::object>("test321");
      EXPECT_TRUE(std::string("test321") == egress_test->name());
      EXPECT_TRUE(egress_test->is_sink());

      builder.make_edge(src, egress_test);
    };

    std::function<void(srf::segment::Builder&)> seg2_init = [](srf::segment::Builder& builder) {
        auto sink =
            builder.make_sink<py::object>("sink", rxcpp::make_observer_dynamic<py::object>([](py::object data) {
                                              // Write to the log
                                              std::cerr << "Got object!" << std::endl;
                                              py::gil_scoped_acquire gil;
                                              py::print(data);
                                          }));

        auto ingress_test = builder.get_ingress<py::object>("test321");
        EXPECT_TRUE(std::string("test321") == ingress_test->name());
        EXPECT_TRUE(ingress_test->is_source());
        builder.make_edge(ingress_test, sink);
    };

    pysrf::Pipeline seg1_pipe;
    pysrf::Pipeline seg2_pipe;

    seg1_pipe.make_segment("TestSegment1", ingress_port_ids, egress_port_ids, seg1_init);
    seg2_pipe.make_segment("TestSegment2", egress_port_ids, ingress_port_ids, seg2_init);

    auto opt1 = std::make_shared<srf::Options>();
    auto opt2 = std::make_shared<srf::Options>();
    opt1->architect_url("127.0.0.1:13337");
    opt1->topology().user_cpuset("0-8");
    opt1->enable_server(true);
    opt1->topology().restrict_gpus(true);
    opt1->config_request("TestSegment1");

    opt2->architect_url("127.0.0.1:13337");
    opt2->topology().user_cpuset("9-16");
    opt2->topology().restrict_gpus(true);
    opt2->config_request("TestSegment2");

    srf::Executor exec1{opt1};
    srf::Executor exec2{opt2};

    exec1.register_pipeline(seg1_pipe.swap());
    exec2.register_pipeline(seg2_pipe.swap());

    py::gil_scoped_release release;
    auto start_1 = boost::fibers::async([&] { exec1.start(); });
    auto start_2 = boost::fibers::async([&] { exec2.start(); });

    start_1.get();
    start_2.get();

    exec1.stop();
    exec2.stop();

    exec1.join();
    exec2.join();
}*/

TEST_F(TestPipeline, DynamicPortConstructionTooManyPorts)
{
    std::string name                                 = "xyz";
    std::function<void(srf::segment::Builder&)> init = [](srf::segment::Builder& builder) {
        std::cerr << "Builder called" << std::endl;
    };

    std::vector<std::vector<std::string>> ingress_id_vec;
    std::vector<std::vector<std::string>> egress_id_vec;

    for (int i = 0; i <= SRF_MAX_INGRESS_PORTS; ++i)
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

    for (int i = 0; i <= SRF_MAX_EGRESS_PORTS; ++i)
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

    std::vector<std::string> too_many_ports = {
        "_01", "_02", "_03", "_04", "_05", "_06", "_07", "_08", "_09", "_10", "_11"};

    for (auto ivec : ingress_id_vec)
    {
        pysrf::Pipeline pipe;
        EXPECT_THROW(pipe.make_segment(name, ivec, too_many_ports, init), std::runtime_error);
    }

    for (auto evec : egress_id_vec)
    {
        pysrf::Pipeline pipe;
        EXPECT_THROW(pipe.make_segment(name, too_many_ports, evec, init), std::runtime_error);
    }
}
