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

#include "mrc/benchmarking/trace_statistics.hpp"

#include "mrc/benchmarking/util.hpp"
#include "mrc/core/watcher.hpp"  // for WatchableEvent

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>  // for pair, make_pair

using nlohmann::json;

namespace mrc::benchmarking {

thread_local std::map<std::string, std::shared_ptr<TraceStatistics>> TraceStatistics::TraceObjectMap{};
std::multimap<std::string, std::shared_ptr<TraceStatistics>> TraceStatistics::TraceObjectMultimap{};
std::recursive_mutex TraceStatistics::s_state_mutex{};

bool TraceStatistics::s_trace_operators              = (std::getenv("MRC_TRACE_OPERATORS") != nullptr);
bool TraceStatistics::s_trace_operators_set_manually = false;
bool TraceStatistics::s_trace_channels               = (std::getenv("MRC_TRACE_CHANNELS") != nullptr);
bool TraceStatistics::s_trace_channels_set_manually  = false;
bool TraceStatistics::s_initialized                  = false;

void TraceStatistics::init()
{
    if (s_initialized)
    {
        return;
    }

    TimeUtil::estimate_steady_clock_delay();
    s_initialized = true;
}

std::shared_ptr<TraceStatistics> TraceStatistics::get_or_create(const std::string& name)
{
    std::lock_guard<std::recursive_mutex> lock(s_state_mutex);

    auto it = TraceObjectMap.find(name);
    if (it == TraceObjectMap.end())
    {
        std::stringstream sstream;
        sstream << name << "_" << std::this_thread::get_id();

        // TraceStatistics constructor is private to force per-thread unique names and to ensure objects are
        //  tracked correctly via the cross thread multi-map.
        auto stats           = std::shared_ptr<TraceStatistics>(new TraceStatistics(name));
        TraceObjectMap[name] = stats;
        TraceObjectMultimap.insert(std::make_pair(name, stats));

        VLOG(5) << "Creating TracerObjectMap entry for " << sstream.str() << " at 0x" << stats.get() << std::endl;
    }

    return TraceObjectMap[name];
}

TraceStatistics::TraceStatistics(const std::string& name) :
  m_name(name),
  m_start_time(TimeUtil::get_delay_compensated_time_point()),
  m_internal_chain_start(m_start_time),
  m_channel_read_start(m_start_time),
  m_channel_write_start(m_start_time),
  m_parent_id(std::this_thread::get_id())
{
    initialize_lookup_tables();
}

const std::multimap<std::string, std::shared_ptr<TraceStatistics>>& TraceStatistics::get_thread_local_results_map()
{
    return TraceObjectMultimap;
}

void TraceStatistics::trace_operators(bool flag, bool sync_immediate)
{
    s_trace_operators              = flag;
    s_trace_operators_set_manually = true;

    if (sync_immediate)
    {
        sync_state();
    }
}

std::tuple<bool, bool> TraceStatistics::trace_operators()
{
    return std::make_pair(s_trace_operators, s_trace_operators_set_manually);
}

void TraceStatistics::trace_channels(bool flag, bool sync_immediate)
{
    s_trace_channels              = flag;
    s_trace_channels_set_manually = true;

    if (sync_immediate)
    {
        sync_state();
    }
}

std::tuple<bool, bool> TraceStatistics::trace_channels()
{
    return std::make_pair(s_trace_channels, s_trace_channels_set_manually);
}

/*
 * @brief return a snapshot in time of the current state. This does race with the writing thread, so it
 * should be considered approximately accurate.
 */
json TraceStatistics::to_json() const
{
    std::size_t total_elapsed_ns =
        TimeUtil::time_resolution_unit_t(TimeUtil::get_current_time_point() - m_start_time).count();

    std::size_t emission_count            = m_emission_count;
    std::size_t receive_count             = m_receive_count;
    std::size_t ch_read_count             = m_channel_sink_reads;
    std::size_t ch_write_count            = m_channel_source_writes;
    std::size_t total_internal_elapsed_ns = m_total_internal_elapsed_ns;
    std::size_t total_ch_read_elapsed_ns  = m_total_ch_read_elapsed_ns;
    std::size_t total_ch_write_elapsed_ns = m_total_ch_write_elapsed_ns;

    double scaling_coef         = total_elapsed_ns * TimeUtil::NsToSec;
    double emissions_per_second = emission_count / scaling_coef;
    double received_per_second  = receive_count / scaling_coef;

    std::size_t average_op_latency_ns       = emission_count > 0 ? total_internal_elapsed_ns / (emission_count) : 0.0;
    std::size_t average_ch_read_latency_ns  = ch_read_count > 0 ? total_ch_read_elapsed_ns / (ch_read_count) : 0.0;
    std::size_t average_ch_write_latency_ns = ch_write_count > 0 ? total_ch_write_elapsed_ns / (ch_write_count) : 0.0;

    return json::object({{"component_emissions_total", emission_count},
                         {"component_emission_rate_seconds", emissions_per_second},
                         {"component_receive_total", receive_count},
                         {"component_received_rate_seconds", received_per_second},
                         {"component_channel_read_total", ch_read_count},
                         {"component_channel_write_total", ch_write_count},
                         {"component_operator_proc_latency_seconds", average_op_latency_ns * TimeUtil::NsToSec},
                         {"component_channel_read_latency_seconds", average_ch_read_latency_ns * TimeUtil::NsToSec},
                         {"component_channel_write_latency_seconds", average_ch_write_latency_ns * TimeUtil::NsToSec},
                         {"component_channel_reads_seconds", total_ch_read_elapsed_ns * TimeUtil::NsToSec},
                         {"component_channel_write_seconds", total_ch_write_elapsed_ns * TimeUtil::NsToSec},
                         {"component_operator_proc_seconds", total_internal_elapsed_ns * TimeUtil::NsToSec},
                         {"component_elapsed_total_seconds", total_elapsed_ns * TimeUtil::NsToSec}});
}

json TraceStatistics::aggregate()
{
    json aggregation = {{"aggregations",
                         {{"components", {{"metrics", json::object()}}},
                          {"metrics",
                           {{"counter",
                             {
                                 {"component_emissions_total", json::array()},
                                 {"component_emission_rate_seconds", json::array()},
                                 {"component_receive_total", json::array()},
                                 {"component_received_rate_seconds", json::array()},
                                 {"component_channel_read_total", json::array()},
                                 {"component_channel_write_total", json::array()},
                                 {"component_operator_proc_latency_seconds", json::array()},
                                 {"component_channel_read_latency_seconds", json::array()},
                                 {"component_channel_write_latency_seconds", json::array()},
                                 {"component_channel_reads_seconds", json::array()},
                                 {"component_channel_write_seconds", json::array()},
                                 {"component_operator_proc_seconds", json::array()},
                             }}}}}},
                        {"metadata",
                         {{"elapsed_time", 0.0},
                          {"internal_elapsed_time", 0.0},
                          {"metric_data_types",
                           {
                               {"component_emissions_total", "ui"},
                               {"component_emission_rate_seconds", "d"},
                               {"component_receive_total", "ui"},
                               {"component_received_rate_seconds", "d"},
                               {"component_channel_read_total", "ui"},
                               {"component_channel_write_total", "ui"},
                               {"component_operator_proc_latency_seconds", "d"},
                               {"component_channel_read_latency_seconds", "d"},
                               {"component_channel_write_latency_seconds", "d"},
                               {"component_channel_reads_seconds", "d"},
                               {"component_channel_write_seconds", "d"},
                               {"component_operator_proc_seconds", "d"},
                               {"component_elapsed_total_seconds", "ui"},
                           }}}}};

    json& counters          = aggregation["aggregations"]["metrics"]["counter"];
    json& component_metrics = aggregation["aggregations"]["components"]["metrics"];
    json& metadata          = aggregation["metadata"];
    json& data_types        = metadata["metric_data_types"];

    json current_object = {};
    std::string current_component;
    std::size_t count = 0;
    for (auto& it : TraceObjectMultimap)
    {
        if (current_component != it.first)
        {
            for (json::iterator current_it = current_object.begin(); current_it != current_object.end(); current_it++)
            {
                auto key   = current_it.key();
                auto value = current_it.value();

                // Prometheus style metric storage -- each metric is stored with entries for each component
                counters[key].push_back({{"labels", {{"component_id", current_component}}}, {"value", value}});

                // Component based metric storage
                component_metrics[current_component][key] = value;
            }

            count                                = 0;
            current_component                    = it.first;
            current_object                       = it.second->to_json();
            component_metrics[current_component] = json::object();

            continue;
        }

        auto sibling = it.second->to_json();

        /*
         * Note: Sanity check for posterity: obj.get<type> will try to return the underlying memory as the
         * specified type, it will not do any kind of conversion. So, if your field contains an int and you
         * make a .get<double> call, you will get back gibberish. The correct way to do the cast would be:
         * static_cast<double>(obj.get<int>)
         */
        for (json::iterator sibling_it = sibling.begin(); sibling_it != sibling.end(); sibling_it++)
        {
            auto key   = sibling_it.key();
            auto value = sibling_it.value();

            if (data_types[key] == "ui")
            {
                current_object[key] = current_object[key].get<std::size_t>() + value.get<std::size_t>();
            }
            else if (data_types[key] == "d")
            {
                current_object[key] = current_object[key].get<double>() + value.get<double>();
            }
            else
            {
                std::stringstream sstream;

                sstream << "Unknown metric data type: " << key << " -- " << data_types[key];
                throw std::runtime_error(sstream.str());
            }
        }

        count++;
    }

    {
        for (json::iterator current_it = current_object.begin(); current_it != current_object.end(); current_it++)
        {
            auto key   = current_it.key();
            auto value = current_it.value();

            // Prometheus style metric storage -- each metric is stored with entries for each component
            counters[key].push_back({{"labels", {{"component_id", current_component}}}, {"value", value}});
            component_metrics[current_component][key] = value;
        }
    }

    return aggregation;
}

void TraceStatistics::channel_read_start()
{
    m_channel_read_start = TimeUtil::get_delay_compensated_time_point();
}

void TraceStatistics::channel_read_end()
{
    auto now     = TimeUtil::get_current_time_point();
    auto elapsed = now > m_channel_read_start ? TimeUtil::time_resolution_unit_t(now - m_channel_write_start)
                                              : TimeUtil::s_minimum_resolution;
    m_total_ch_read_elapsed_ns += elapsed.count();
    m_channel_sink_reads += 1;
}

void TraceStatistics::channel_write_start()
{
    m_channel_write_start = TimeUtil::get_delay_compensated_time_point();
}

void TraceStatistics::channel_write_end()
{
    auto now     = TimeUtil::get_current_time_point();
    auto elapsed = now > m_channel_write_start ? TimeUtil::time_resolution_unit_t(now - m_channel_write_start)
                                               : TimeUtil::s_minimum_resolution;
    m_total_ch_write_elapsed_ns += elapsed.count();
    m_channel_source_writes += 1;
    m_internal_chain_start = TimeUtil::get_delay_compensated_time_point();
}

void TraceStatistics::clear()
{
    m_emission_count            = 0;
    m_receive_count             = 0;
    m_channel_sink_reads        = 0;
    m_channel_source_writes     = 0;
    m_total_ch_read_elapsed_ns  = 0;
    m_total_ch_write_elapsed_ns = 0;
    m_total_internal_elapsed_ns = 0;
    m_start_time                = TimeUtil::get_delay_compensated_time_point();
}

void TraceStatistics::initialize_lookup_tables()
{
    std::function<void(void)> emit_           = []() {};
    std::function<void(void)> receive_        = []() {};
    std::function<void(void)> ch_read_start_  = []() {};
    std::function<void(void)> ch_read_end_    = []() {};
    std::function<void(void)> ch_write_start_ = []() {};
    std::function<void(void)> ch_write_end_   = []() {};

    if (TraceStatistics::s_trace_operators)
    {
        init();
        emit_    = [this]() { emit(); };
        receive_ = [this]() { receive(); };
    }

    if (TraceStatistics::s_trace_channels)
    {
        init();
        ch_read_start_ = [this]() { channel_read_start(); };
        ch_read_end_   = [this]() { channel_read_end(); };

        ch_write_start_ = [this]() { channel_write_start(); };
        ch_write_end_   = [this]() { channel_write_end(); };
    }

    m_entry_lookup_table[static_cast<std::size_t>(WatchableEvent::sink_on_data)]  = receive_;
    m_entry_lookup_table[static_cast<std::size_t>(WatchableEvent::channel_read)]  = ch_read_start_;
    m_entry_lookup_table[static_cast<std::size_t>(WatchableEvent::channel_write)] = ch_write_start_;

    m_exit_lookup_table[static_cast<std::size_t>(WatchableEvent::sink_on_data)]  = emit_;
    m_exit_lookup_table[static_cast<std::size_t>(WatchableEvent::channel_read)]  = ch_read_end_;
    m_exit_lookup_table[static_cast<std::size_t>(WatchableEvent::channel_write)] = ch_write_end_;
}

void TraceStatistics::reset()
{
    std::lock_guard<std::recursive_mutex> lock(s_state_mutex);

    s_trace_operators              = false;
    s_trace_operators_set_manually = false;

    s_trace_channels              = false;
    s_trace_channels_set_manually = false;

    sync_state();
    for (auto& mm_iter : TraceObjectMultimap)
    {
        mm_iter.second->clear();
    }
}

void TraceStatistics::sync_state()
{
    std::lock_guard<std::recursive_mutex> lock(s_state_mutex);

    TraceStatistics::s_trace_operators =
        s_trace_operators_set_manually ? s_trace_operators : (std::getenv("MRC_TRACE_OPERATORS") != nullptr);
    TraceStatistics::s_trace_channels =
        s_trace_channels_set_manually ? s_trace_channels : (std::getenv("MRC_TRACE_CHANNELS") != nullptr);

    for (auto& mm_iter : TraceObjectMultimap)
    {
        mm_iter.second->initialize_lookup_tables();
    }
}

void TraceStatistics::emit()
{
    auto now     = TimeUtil::get_current_time_point();
    auto elapsed = now > m_internal_chain_start ? TimeUtil::time_resolution_unit_t(now - m_internal_chain_start)
                                                : TimeUtil::s_minimum_resolution;
    m_emission_count++;
    m_total_internal_elapsed_ns += elapsed.count();

    /* If we're an internal node, this will be re-set on the next receive call; otherwise, we'll use emit->emit timings
     *  to produce a sane metric to report for source node operator latency.
     */
    m_internal_chain_start = TimeUtil::get_delay_compensated_time_point();
}

void TraceStatistics::on_entry(const WatchableEvent& e, const void* data)
{
    m_entry_lookup_table[static_cast<std::size_t>(e)]();
}

void TraceStatistics::on_exit(const WatchableEvent& e, bool rc, const void* data)
{
    m_exit_lookup_table[static_cast<std::size_t>(e)]();
}

std::thread::id TraceStatistics::parent_id() const
{
    return m_parent_id;
}

void TraceStatistics::receive()
{
    m_internal_chain_start = TimeUtil::get_delay_compensated_time_point();
    m_receive_count++;
}

}  // namespace mrc::benchmarking
