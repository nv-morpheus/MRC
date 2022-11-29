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

#pragma once

#include "../test_segment.hpp"

#include "mrc/benchmarking/segment_watcher.hpp"
#include "mrc/benchmarking/tracer.hpp"

class LatencyBenchmarkTests : public ::testing::Test
{
  public:
    using latency_tracer_t = benchmarking::TracerEnsemble<std::size_t, benchmarking::LatencyTracer>;
    using tracer_type_t    = latency_tracer_t;
    using data_type_t      = std::shared_ptr<latency_tracer_t>;

    void SetUp() override
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<> dist(10, 100);

        m_iterations = dist(generator);
        auto init    = [this](segment::Builder& segment) {
            std::string src_name  = "nsrc";
            std::string int_name  = "n1";
            std::string sink_name = "nsink";

            auto src =
                segment.make_source<data_type_t>(src_name, m_watcher->template create_rx_tracer_source<true>(src_name));

            auto internal_idx = m_watcher->get_or_create_node_entry(int_name);
            auto internal     = segment.make_node<data_type_t, data_type_t>(
                int_name,
                rxcpp::operators::tap([internal_idx](data_type_t tracer) { tracer->receive(internal_idx); }),
                rxcpp::operators::map([](data_type_t tracer) { return tracer; }),
                rxcpp::operators::tap([internal_idx](data_type_t tracer) { tracer->emit(internal_idx); }));
            segment.make_edge(src, internal);

            auto sink_idx = m_watcher->get_or_create_node_entry(sink_name);
            auto sink     = segment.make_sink<data_type_t>(
                sink_name, m_watcher->create_tracer_sink_lambda(sink_name, [](tracer_type_t& data) {}));
            segment.make_edge(internal, sink);
        };

        auto pipeline = pipeline::make_pipeline();
        auto segment  = pipeline->make_segment("bench_segment", init);

        std::shared_ptr<Executor> executor = std::make_shared<Executor>();
        executor->register_pipeline(std::move(pipeline));

        m_watcher = std::make_unique<benchmarking::SegmentWatcher<tracer_type_t>>(executor);
    }

    void TearDown() override
    {
        m_watcher->shutdown();
    }

    std::unique_ptr<benchmarking::SegmentWatcher<tracer_type_t>> m_watcher;
    std::set<std::string> m_components = {"nsrc", "n1", "nsink"};
    std::size_t m_iterations;
};

class ThroughputBenchmarkTests : public ::testing::Test
{
  public:
    using throughput_tracer_t = benchmarking::TracerEnsemble<std::size_t, benchmarking::ThroughputTracer>;
    using tracer_type_t       = throughput_tracer_t;
    using data_type_t         = std::shared_ptr<throughput_tracer_t>;

    void SetUp() override
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<> dist(10, 100);

        m_iterations = dist(generator);
        auto init    = [this](segment::Builder& segment) {
            std::string src_name  = "nsrc";
            std::string int_name  = "n1";
            std::string sink_name = "nsink";

            auto src = segment.make_source<data_type_t>(src_name,
                                                        m_watcher->template create_rx_tracer_source<false>(src_name));

            auto internal_idx = m_watcher->get_or_create_node_entry(int_name);
            auto internal     = segment.make_node<data_type_t, data_type_t>(
                int_name,
                rxcpp::operators::tap([internal_idx](data_type_t tracer) { tracer->receive(internal_idx); }),
                rxcpp::operators::map([](data_type_t tracer) { return tracer; }),
                rxcpp::operators::tap([internal_idx](data_type_t tracer) { tracer->emit(internal_idx); }));
            segment.make_edge(src, internal);

            auto sink_idx = m_watcher->get_or_create_node_entry(sink_name);
            auto sink     = segment.make_sink<data_type_t>(
                sink_name, m_watcher->create_tracer_sink_lambda(sink_name, [](tracer_type_t& data) {}));
            segment.make_edge(internal, sink);
        };

        auto pipeline = pipeline::make_pipeline();
        auto segment  = pipeline->make_segment("bench_segment", init);

        std::shared_ptr<Executor> executor = std::make_shared<Executor>();
        executor->register_pipeline(std::move(pipeline));

        m_watcher = std::make_unique<benchmarking::SegmentWatcher<tracer_type_t>>(executor);
    }

    void TearDown() override
    {
        m_watcher->shutdown();
    }

    std::unique_ptr<benchmarking::SegmentWatcher<tracer_type_t>> m_watcher;
    std::set<std::string> m_components = {"nsrc", "n1", "nsink"};
    std::size_t m_iterations;
};
