/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/benchmarking/tracer.hpp"
#include "mrc/benchmarking/util.hpp"

#include <benchmark/benchmark.h>
#include <rxcpp/rx.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using namespace mrc::benchmarking;

namespace defs {
using clock_t   = std::chrono::steady_clock;
using time_pt_t = std::chrono::time_point<clock_t>;

static size_t object_count = 1e6;

static defs::time_pt_t tracing_start_ns{clock_t::now()};
static std::size_t elapsed_total_ns{0};
static size_t count{0};
static std::vector<std::shared_ptr<TracerBase>> m_tracers(object_count);
static double m_mean_latency = 0.0;

// Used with benchmark::DoNotOptimize to make sure we have guaranteed work.
std::random_device device;
std::mt19937 generator(device());
std::uniform_real_distribution<> real_dist(0.1, 3.14);

void reset()
{
    count            = 0;
    elapsed_total_ns = 0;
    m_tracers.clear();

    TimeUtil::init();
}
}  // namespace defs

struct DataPayload
{
    DataPayload() = default;

    std::size_t m_data_payload_counter{0};
};

struct DataObject
{
    DataObject() : m_payload{std::make_shared<DataPayload>()} {};

    std::size_t m_data_object_counter{0};

    std::shared_ptr<DataPayload> m_payload;
};

void map_latency()
{
    auto ints = rxcpp::observable<>::create<double>([&](rxcpp::subscriber<double> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_current_time_point();
        for (std::size_t i = 0; i < defs::object_count; ++i)
        {
            subscriber.on_next(static_cast<double>(i));
        }
        subscriber.on_completed();
        defs::elapsed_total_ns += (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
    });

    ints.map([](double value) {
            double new_value;
            benchmark::DoNotOptimize(new_value = defs::real_dist(defs::generator));

            return new_value;
        })
        .subscribe(
            [](double value) {
                defs::count++;
            },
            []() {});
}

void tap_latency()
{
    auto ints = rxcpp::observable<>::create<int>([&](rxcpp::subscriber<int> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_current_time_point();
        for (std::size_t i = 0; i < defs::object_count; ++i)
        {
            subscriber.on_next(i);
        }
        subscriber.on_completed();
        defs::elapsed_total_ns += (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
    });

    ints.tap([](int value) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
        })
        .subscribe(
            [](int value) {
                defs::count++;
            },
            []() {});
}

void debug_noop_maptap_sharedptr_latency(const std::size_t packet_count)
{
    using data_type_t = std::shared_ptr<DataObject>;

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            auto data_object = std::make_shared<DataObject>();
            subscriber.on_next(data_object);
        }
        subscriber.on_completed();
    });

    ints.map([](data_type_t data) {
            return data;
        })
        .map([](data_type_t data) {
            data->m_data_object_counter++;
            data->m_payload->m_data_payload_counter++;
            return data;
        })
        .map([](data_type_t data) {
            return data;
        })
        .subscribe(
            [](data_type_t data) {
                defs::count += 1;
                if (defs::count == defs::object_count)
                {
                    defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
                }
            },
            []() {});
}

void debug_noop_tap_sharedptr_latency(const std::size_t packet_count)
{
    using data_type_t = std::shared_ptr<DataObject>;

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            auto data_object = std::make_shared<DataObject>();
            subscriber.on_next(data_object);
        }
        subscriber.on_completed() /**/;
    });

    ints.tap([](data_type_t data) {})
        .map([](data_type_t data) {
            data->m_data_object_counter++;
            data->m_payload->m_data_payload_counter++;
            return data;
        })
        .tap([](data_type_t data) {})
        .subscribe(
            [](data_type_t data) {
                defs::count += 1;
                if (defs::count == defs::object_count)
                {
                    defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
                }
            },
            []() {});
}

void debug_maptap_sharedptr_latency(const std::size_t packet_count)
{
    using data_type_t = std::shared_ptr<DataObject>;

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            auto data_object = std::make_shared<DataObject>();
            subscriber.on_next(data_object);
        }
        subscriber.on_completed() /**/;
    });

    // Use a map instead of a tap. Prior to RxCPP 4.1 this was substantially faster than using taps.
    ints.map([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
            return data;
        })
        .map([](data_type_t data) {
            data->m_data_object_counter++;
            data->m_payload->m_data_payload_counter++;
            return data;
        })
        .map([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
            return data;
        })
        .subscribe(
            [](data_type_t data) {
                defs::count += 1;
                if (defs::count == defs::object_count)
                {
                    defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
                }
            },
            []() {});
}

void debug_tap_sharedptr_latency(const std::size_t packet_count)
{
    using data_type_t = std::shared_ptr<DataObject>;

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            auto data_object = std::make_shared<DataObject>();
            subscriber.on_next(data_object);
        }
        subscriber.on_completed() /**/;
    });

    ints.tap([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
        })
        .map([](data_type_t data) {
            data->m_data_object_counter++;
            data->m_payload->m_data_payload_counter++;
            return data;
        })
        .tap([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
        })
        .subscribe(
            [](data_type_t data) {
                defs::count += 1;
                if (defs::count == defs::object_count)
                {
                    defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
                }
            },
            []() {});
}

void debug_tap_sharedptr_dynamic_observer_latency(const std::size_t packet_count)
{
    using latency_tracer_t = TracerEnsemble<std::size_t, LatencyTracer>;

    using data_type_t = std::shared_ptr<latency_tracer_t>;

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> s) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            auto tracer = std::make_shared<latency_tracer_t>(3);
            tracer->recv_hop_id(0);
            tracer->reset();
            s.on_next(tracer);
        }
        s.on_completed() /**/;
    });

    auto tap_1 = rxcpp::operators::tap([](data_type_t data) {
        benchmark::DoNotOptimize(defs::real_dist(defs::generator));
    });
    auto map_1 = rxcpp::operators::map([](data_type_t data) {
        return data;
    });
    auto tap_2 = rxcpp::operators::tap([](data_type_t data) {
        benchmark::DoNotOptimize(defs::real_dist(defs::generator));
    });

    auto body_observable = ints | tap_1 | map_1 | tap_2;

    auto dyn_observer = rxcpp::make_observer_dynamic<data_type_t>(
        [](data_type_t data) {
            data->emit(0);
            defs::count++;
            defs::m_tracers.push_back(data);
        },
        [](rxcpp::util::error_ptr error) {
            std::cerr << "Error occurred" << std::endl;
        },
        []() {
            defs::count            = defs::m_tracers.size();
            defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();

            auto trace_aggregator = std::make_shared<TraceAggregator<latency_tracer_t>>();
            trace_aggregator->process_tracer_data(defs::m_tracers,
                                                  defs::elapsed_total_ns / 1e9,
                                                  3,
                                                  {{0, "src"}, {1, "n1"}, {2, "sink"}});
            auto jsd =
                trace_aggregator->to_json()["aggregations"]["metrics"]["counter"]["component_latency_seconds_mean"][0];
            defs::m_mean_latency = jsd["value"].get<double>();
        });

    body_observable.subscribe(dyn_observer);
}

void sharedptr_nocreate_latency(const std::size_t packet_count)
{
    using data_type_t = std::shared_ptr<DataObject>;
    auto data_object  = std::make_shared<DataObject>();

    auto ints = rxcpp::observable<>::create<data_type_t>([&](rxcpp::subscriber<data_type_t> subscriber) {
        defs::tracing_start_ns = TimeUtil::get_delay_compensated_time_point();
        for (auto i = 0; i < packet_count; ++i)
        {
            subscriber.on_next(data_object);
        }
        subscriber.on_completed() /**/;
    });

    ints.tap([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
        })
        .map([](data_type_t data) {
            data->m_data_object_counter++;
            data->m_payload->m_data_payload_counter++;
            return data;
        })
        .tap([](data_type_t data) {
            benchmark::DoNotOptimize(defs::real_dist(defs::generator));
        })
        .subscribe(
            [](data_type_t data) {
                defs::count += 1;
                if (defs::count == defs::object_count)
                {
                    defs::elapsed_total_ns = (TimeUtil::get_current_time_point() - defs::tracing_start_ns).count();
                }
            },
            []() {});
}

static void rx_map_latency_raw(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        map_latency();
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_tap_latency_raw(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        tap_latency();
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_sharedptr_nocreate_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        sharedptr_nocreate_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_debug_noop_tap_sharedptr_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        debug_noop_tap_sharedptr_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_debug_noop_maptap_sharedptr_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        debug_noop_maptap_sharedptr_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_debug_tap_sharedptr_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        debug_tap_sharedptr_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void rx_debug_tap_sharedptr_dynamic_observer_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        debug_tap_sharedptr_dynamic_observer_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = defs::m_mean_latency;
}

static void rx_debug_maptap_sharedptr_latency(benchmark::State& state)
{
    for (auto _ : state)
    {
        defs::reset();
        debug_maptap_sharedptr_latency(defs::object_count);
    }

    state.counters["elapsed_seconds"]         = defs::elapsed_total_ns * TimeUtil::NsToSec;
    state.counters["count"]                   = defs::count;
    state.counters["average_latency_seconds"] = static_cast<double>(defs::elapsed_total_ns * TimeUtil::NsToSec) /
                                                defs::object_count;
}

static void chrono_steady_clock_now(benchmark::State& state)
{
    defs::reset();
    for (auto _ : state)
    {
        TimeUtil::time_pt_t now;
        benchmark::DoNotOptimize(now = std::chrono::steady_clock::now());
    }
    state.counters["time_util_estimate"] = TimeUtil::s_mean_steady_clock_call_unit_delay;
}

static void chrono_highres_clock_now(benchmark::State& state)
{
    defs::reset();
    for (auto _ : state)
    {
        TimeUtil::time_pt_t now;
        benchmark::DoNotOptimize(now = std::chrono::steady_clock::now());
    }
    state.counters["time_util_estimate"] = TimeUtil::s_mean_steady_clock_call_unit_delay;
}

BENCHMARK(chrono_steady_clock_now)->UseRealTime();
BENCHMARK(chrono_highres_clock_now)->UseRealTime();
BENCHMARK(rx_map_latency_raw)->UseRealTime();
BENCHMARK(rx_tap_latency_raw)->UseRealTime();
BENCHMARK(rx_sharedptr_nocreate_latency)->UseRealTime();
BENCHMARK(rx_debug_noop_tap_sharedptr_latency)->UseRealTime();
BENCHMARK(rx_debug_noop_maptap_sharedptr_latency)->UseRealTime();
BENCHMARK(rx_debug_tap_sharedptr_latency)->UseRealTime();
BENCHMARK(rx_debug_tap_sharedptr_dynamic_observer_latency)->UseRealTime();
BENCHMARK(rx_debug_maptap_sharedptr_latency)->UseRealTime();
