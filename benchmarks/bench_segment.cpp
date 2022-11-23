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

#include "mrc/benchmarking/segment_watcher.hpp"
#include "mrc/benchmarking/tracer.hpp"
#include "mrc/benchmarking/util.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/executor.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <benchmark/benchmark.h>
#include <boost/hana/if.hpp>
#include <nlohmann/json.hpp>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/operators/rx-tap.hpp>
#include <rxcpp/rx.hpp>  // IWYU pragma: keep
#include <rxcpp/sources/rx-iterate.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mrc;
using namespace mrc::benchmarking;
using namespace std::literals::chrono_literals;

/**
 * Benchmarks are currently setup in a way that allows for testing of raw channel / rx operator latency and to
 * illustrate how the watcher objects can be used.
 */

/**
 * @brief Given a TraceAggregator object, extract metrics of interest and add them to the benchmarking state object.
 * @param aggregator
 * @param state
 */
static void add_state_counters(std::shared_ptr<TraceAggregatorBase> aggregator, ::benchmark::State& state)
{
    using nlohmann::json;
    auto json_data = aggregator->to_json();
    auto& counters = json_data["aggregations"]["metrics"]["counter"];
    auto& metadata = json_data["metadata"];

    // std::cerr << json_data.dump(2) << std::endl;
    state.counters["tracers_total"] = metadata["tracer_count"].get<std::size_t>();
    state.counters["segment_mean_throughput"] =
        metadata["elapsed_time"].get<double>() / metadata["tracer_count"].get<std::size_t>();
    state.counters["segment_elapsed_seconds"] = metadata["elapsed_time"].get<double>();
    std::set<std::string> trace_types         = {"component_latency_seconds_mean", "component_mean_throughput"};
    for (auto counter = counters.begin(); counter != counters.end(); ++counter)
    {
        if (trace_types.find(counter.key()) != trace_types.end())
        {
            for (auto& metric : counter.value())
            {
                std::stringstream counter_id;
                json& labels = metric["labels"];

                if (labels["type"] == "operator")
                {
                    counter_id << labels["source_name"];
                    state.counters[counter_id.str()] = metric["value"];
                }
                else if (labels["type"] == "channel")
                {
                    counter_id << labels["source_name"] << "->" << labels["dest_name"];
                    state.counters[counter_id.str()] = metric["value"];
                }
            }
        }
    }
}

/**
 * @brief A basic 3 layer segment, src->intermediate->sink
 * @tparam TracerTypeT Should be a type of tracer Ensemble that indicates the collection of tracers passed through the
 * network.
 * @tparam OneAtATimeV Indicates whether or not tracers should be sent through the segment one at a time. This can be
 * used to test the maximum raw throughput of each component.
 */
template <class TracerTypeT>
class ManualRxcppFixture : public benchmark::Fixture
{
  public:
    using clock_t       = std::chrono::steady_clock;
    using tracer_type_t = TracerTypeT;
    using data_type_t   = std::shared_ptr<tracer_type_t>;

    static constexpr std::size_t PacketCount = 1e4;

    void reset()
    {
        m_count      = 0;
        m_elapsed_ns = 0;
        m_marker     = clock_t::now();
        m_tracers.clear();
    }

    void SetUp(const ::benchmark::State& state) override
    {
        TimeUtil::estimate_steady_clock_delay();
        auto ints = rxcpp::observable<>::create<data_type_t>([](rxcpp::subscriber<data_type_t> subscriber) {
            for (auto i = 0; i < PacketCount; ++i)
            {
                auto tracer = std::make_shared<TracerTypeT>(3);
                tracer->recv_hop_id(0);
                tracer->reset();
                subscriber.on_next(tracer);
            }
            subscriber.on_completed() /**/;
        });

        auto tap_1 = rxcpp::operators::tap([](data_type_t data) {});
        auto map_1 = rxcpp::operators::map([](data_type_t data) { return data; });
        auto tap_2 = rxcpp::operators::tap([](data_type_t data) {});

        m_observable = ints | tap_1 | map_1 | tap_2;

        m_observer = rxcpp::make_observer_dynamic<data_type_t>(
            [this](data_type_t data) {
                data->emit(0);
                m_tracers.push_back(data);
            },
            [](rxcpp::util::error_ptr error) { std::cerr << "Error occurred" << std::endl; },
            [this]() {
                m_count      = m_tracers.size();
                m_elapsed_ns = (clock_t::now() - this->m_marker).count();

                auto trace_aggregator = std::make_shared<TraceAggregator<TracerTypeT>>();
                trace_aggregator->process_tracer_data(
                    m_tracers, m_elapsed_ns / 1e9, 3, {{0, "src"}, {1, "n1"}, {2, "sink"}});
                auto jsd = trace_aggregator
                               ->to_json()["aggregations"]["metrics"]["counter"]["component_latency_seconds_mean"][0];
                m_mean_latency = jsd["value"].template get<double>();
            });
    }

    void TearDown(const ::benchmark::State& state) override {}

    rxcpp::observable<data_type_t> m_observable;
    rxcpp::observer<data_type_t> m_observer;
    std::size_t m_count{0};
    std::size_t m_elapsed_ns{0};
    double m_mean_latency{0.0};
    std::chrono::steady_clock::time_point m_marker;
    std::vector<std::shared_ptr<TracerBase>> m_tracers;
};

/**
 * @brief A basic 3 layer segment, src->intermediate->sink
 * @tparam TracerTypeT Should be a type of tracer Ensemble that indicates the collection of tracers passed through the
 * network.
 * @tparam OneAtATimeV Indicates whether or not tracers should be sent through the segment one at a time. This can be
 * used to test the maximum raw throughput of each component.
 */
template <class TracerTypeT, bool OneAtATimeV>
class SimpleEmitReceiveFixture : public benchmark::Fixture
{
  public:
    using tracer_type_t = TracerTypeT;
    using data_type_t   = std::shared_ptr<tracer_type_t>;

    void SetUp(const ::benchmark::State& state) override
    {
        TimeUtil::estimate_steady_clock_delay();
        auto init = [this](segment::Builder& segment) {
            std::string src_name  = "nsrc";
            std::string int_name  = "n1";
            std::string sink_name = "nsink";

            auto src = segment.make_source<data_type_t>(
                src_name, m_watcher->template create_rx_tracer_source<OneAtATimeV>(src_name));

            auto internal_idx = m_watcher->get_or_create_node_entry(int_name);
            auto internal     = segment.make_node<data_type_t, data_type_t>(
                int_name,
                m_watcher->create_tracer_receive_tap(int_name),
                rxcpp::operators::map([](data_type_t tracer) { return tracer; }),
                m_watcher->create_tracer_emit_tap(int_name));
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

        m_watcher = std::make_unique<SegmentWatcher<tracer_type_t>>(executor);
    }

    void TearDown(const ::benchmark::State& state) override
    {
        m_watcher->shutdown();
    }

    std::unique_ptr<SegmentWatcher<tracer_type_t>> m_watcher;
};

/**
 * A multi-step linear segment.
 * @tparam TracerTypeT Should be a type of tracer Ensemble that indicates the collection of tracers passed through the
 * network.
 * @tparam OneAtATimeV Indicates whether or not tracers should be sent through the segment one at a time. This can be
 * used to test the maximum raw throughput of each component.
 * @tparam InternalNodesV Number of internal stages the segment should have add.
 */
template <class TracerTypeT, bool OneAtATimeV, std::size_t InternalNodesV>
class LongEmitReceiveFixture : public benchmark::Fixture
{
  public:
    using tracer_type_t = TracerTypeT;  // TracerEnsemble<std::size_t, LatencyTracer<4>>;
    using data_type_t   = std::shared_ptr<tracer_type_t>;

    void SetUp(const ::benchmark::State& state) override
    {
        TimeUtil::estimate_steady_clock_delay();
        auto init = [this](segment::Builder& segment) {
            std::string src_name  = "nsrc";
            std::string sink_name = "nsink";

            auto src = segment.make_source<data_type_t>(
                src_name, m_watcher->template create_rx_tracer_source<OneAtATimeV>(src_name));

            std::shared_ptr<segment::ObjectProperties> last_node = src;

            for (auto i = 0; i < InternalNodesV; ++i)
            {
                auto int_name     = "n" + std::to_string(i);
                auto internal_idx = m_watcher->get_or_create_node_entry(int_name);
                auto internal     = segment.make_node<data_type_t, data_type_t>(
                    int_name,
                    m_watcher->create_tracer_receive_tap(int_name),
                    rxcpp::operators::map([](data_type_t tracer) { return tracer; }),
                    m_watcher->create_tracer_emit_tap(int_name));

                segment.make_dynamic_edge<data_type_t>(last_node, internal);
                last_node = internal;
            }

            auto sink_idx = m_watcher->get_or_create_node_entry(sink_name);
            auto sink     = segment.make_sink<data_type_t>(
                sink_name, m_watcher->create_tracer_sink_lambda(sink_name, [](tracer_type_t& data) {}));

            segment.make_dynamic_edge<data_type_t>(last_node, sink);
        };

        auto pipeline = pipeline::make_pipeline();
        auto segment  = pipeline->make_segment("bench_segment", init);

        std::shared_ptr<Executor> executor = std::make_shared<Executor>();
        executor->register_pipeline(std::move(pipeline));

        m_watcher = std::make_unique<SegmentWatcher<tracer_type_t>>(executor);
    }

    void TearDown(const ::benchmark::State& state) override
    {
        m_watcher->shutdown();
    }

    std::unique_ptr<SegmentWatcher<tracer_type_t>> m_watcher;
};

/** Latency **/
constexpr std::size_t InternalNodeCount = 10;

using latency_tracer_t = TracerEnsemble<std::size_t, LatencyTracer>;
class RxcppManualLatency : public ManualRxcppFixture<latency_tracer_t>
{};

class SegmentComponentRawLatency : public SimpleEmitReceiveFixture<latency_tracer_t, true>
{};

class SegmentComponentLatency : public SimpleEmitReceiveFixture<latency_tracer_t, false>
{};

/** Throughput **/
using throughput_tracer_t = TracerEnsemble<std::size_t, ThroughputTracer>;
class SegmentRawThroughput : public SimpleEmitReceiveFixture<throughput_tracer_t, true>
{};
class SegmentThroughput : public SimpleEmitReceiveFixture<throughput_tracer_t, false>
{};

using latency_tracer_2_t = TracerEnsemble<std::size_t, LatencyTracer>;
class SegmentLongComponentRawLatency : public LongEmitReceiveFixture<latency_tracer_2_t, true, InternalNodeCount>
{};

using throughput_tracer_2_t = TracerEnsemble<std::size_t, ThroughputTracer>;
class SegmentLongComponentRawThroughput : public LongEmitReceiveFixture<throughput_tracer_2_t, true, InternalNodeCount>
{};

// NOLINTNEXTLINE
BENCHMARK_F(RxcppManualLatency, rxcpp_manual_latency)(benchmark::State& state)
{
    for (auto _ : state)
    {
        this->reset();
        m_observable.subscribe(m_observer);
    }

    state.counters["elapsed_seconds"]         = m_elapsed_ns * 1 / 1e9;
    state.counters["tracers_total"]           = m_count;
    state.counters["average_latency_seconds"] = m_mean_latency;
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentComponentRawLatency, component_latency_raw)(benchmark::State& state)
{
    m_watcher->tracer_count(1e3);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentComponentLatency, component_latency)(benchmark::State& state)
{
    m_watcher->tracer_count(1e4);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentRawThroughput, component_throughput_raw)(benchmark::State& state)
{
    m_watcher->tracer_count(1e4);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentThroughput, component_throughput)(benchmark::State& state)
{
    m_watcher->tracer_count(1e4);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentLongComponentRawThroughput, long_pipeline_component_throughput)(benchmark::State& state)
{
    m_watcher->tracer_count(1e4);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}

// NOLINTNEXTLINE
BENCHMARK_F(SegmentLongComponentRawLatency, long_pipeline_component_latency)(benchmark::State& state)
{
    m_watcher->tracer_count(1e4);
    for (auto _ : state)
    {
        m_watcher->reset();
        m_watcher->trace_until_notified();
    }
    add_state_counters(m_watcher->aggregate_tracers(), state);
}
