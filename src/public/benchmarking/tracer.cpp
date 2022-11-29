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

#include "mrc/benchmarking/tracer.hpp"

#include "mrc/benchmarking/util.hpp"

#include <nlohmann/json.hpp>

#include <mutex>

namespace mrc::benchmarking {

using nlohmann::json;

/** Tracer Base **/
TracerBase::TracerBase(const std::size_t max_nodes) : m_max_nodes(max_nodes) {}

std::size_t TracerBase::max_nodes() const
{
    return m_max_nodes;
}

/** Latency Tracer **/
LatencyTracer::LatencyTracer(std::size_t max_nodes) : TracerBase(max_nodes), m_latencies(max_nodes * max_nodes, 0) {}

void LatencyTracer::aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, json& j)

{
    json& metrics           = j["aggregations"]["metrics"]["counter"];
    json& component_metrics = j["aggregations"]["components"]["metrics"];
    json& metadata          = j["metadata"];
    json& id_map            = metadata["id_map"];

    std::size_t node_count = metadata["node_count"];

    /*
     * Aggregate latency counters
     */
    std::vector<double> latencies(node_count * node_count, 0.0);
    for (auto& _tracer : tracers)
    {
        auto tracer = std::dynamic_pointer_cast<LatencyTracer>(_tracer);
        CHECK(tracer->max_nodes() == node_count);

        for (std::size_t emit_idx = 0; emit_idx < node_count; ++emit_idx)
        {
            for (std::size_t recv_idx = 0; recv_idx < node_count; ++recv_idx)
            {
                std::size_t offset = emit_idx * node_count + recv_idx;
                latencies[offset] += tracer->m_latencies[offset];
            }
        }
    }

    metrics["component_latency_seconds_mean"] = json::array();
    json& latency_counters_mean               = metrics["component_latency_seconds_mean"];

    /*
     * Convert nano-seconds to seconds compute mean latency values.
     */
    for (std::size_t emit_idx = 0; emit_idx < node_count; ++emit_idx)
    {
        for (std::size_t recv_idx = 0; recv_idx < node_count; ++recv_idx)
        {
            std::size_t offset          = emit_idx * node_count + recv_idx;
            double latency_mean_seconds = ((latencies[offset] * TimeUtil::NsToSec) / tracers.size());

            const std::string cmpt_type = (emit_idx == recv_idx ? "operator" : "channel");
            if (latency_mean_seconds > 0.0)
            {
                latency_counters_mean.push_back(
                    {{"labels",
                      {{"type", cmpt_type},
                       {"source_name",
                        id_map.contains(std::to_string(emit_idx)) ? id_map[std::to_string(emit_idx)] : ""},
                       {"dest_name",
                        id_map.contains(std::to_string(recv_idx)) ? id_map[std::to_string(recv_idx)] : ""}}},
                     {"value", latency_mean_seconds}});
            }
        }
    }
}

void LatencyTracer::emit(const std::size_t emit_id,
                         const std::size_t recv_id,
                         const time_pt_t& emit_ts,
                         const time_pt_t& recv_ts,
                         const time_unit_ns_t& elapsed)
{
    auto offset = emit_id * m_max_nodes + emit_id;
    m_latencies[offset] += elapsed.count();
}

std::string LatencyTracer::name_id() const
{
    return tracer_name;
}

void LatencyTracer::receive(const std::size_t emit_id,
                            const std::size_t recv_id,
                            const time_pt_t& emit_ts,
                            const time_pt_t& recv_ts,
                            const time_unit_ns_t& elapsed)
{
    auto offset = emit_id * m_max_nodes + recv_id;
    m_latencies[offset] += elapsed.count();
}

/** Throughput tracer **/
ThroughputTracer::ThroughputTracer(std::size_t max_nodes) :
  TracerBase(max_nodes),
  m_latencies(max_nodes * max_nodes, 0),
  m_received_by(max_nodes, 0),
  m_emitted_by(max_nodes, 0)
{}

void ThroughputTracer::aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, json& j)
{
    json& metrics  = j["aggregations"]["metrics"]["counter"];
    json& metadata = j["metadata"];
    json& id_map   = metadata["id_map"];

    std::size_t node_count = metadata["node_count"];

    std::vector<double> processed(node_count, 0.0);
    std::vector<double> latencies(node_count * node_count, 0.0);
    for (auto& _tracer : tracers)
    {
        auto tracer = std::dynamic_pointer_cast<ThroughputTracer>(_tracer);
        CHECK(tracer->max_nodes() == node_count);

        for (std::size_t emit_idx = 0; emit_idx < node_count; ++emit_idx)
        {
            processed[emit_idx] += tracer->m_emitted_by[emit_idx];
            for (std::size_t recv_idx = 0; recv_idx < node_count; ++recv_idx)
            {
                std::size_t offset = emit_idx * node_count + recv_idx;
                latencies[offset] += tracer->m_latencies[offset];
            }
        }
    }

    json& tracer_processed_total  = metrics["component_processed_tracers_total"];
    json& tracer_throughput_total = metrics["component_mean_throughput"];
    for (std::size_t emit_idx = 0; emit_idx < node_count; emit_idx++)
    {
        for (std::size_t recv_idx = 0; recv_idx < node_count; recv_idx++)
        {
            std::size_t offset           = emit_idx * node_count + recv_idx;
            double total_elapsed_seconds = latencies[offset] * TimeUtil::NsToSec;
            std::string cmpt_type        = (emit_idx == recv_idx ? "operator" : "channel");

            if (total_elapsed_seconds > 0.0)
            {
                tracer_processed_total.push_back(
                    {{"labels",
                      {{"type", cmpt_type},
                       {"source_name",
                        id_map.contains(std::to_string(emit_idx)) ? id_map[std::to_string(emit_idx)] : ""},
                       {"dest_name",
                        id_map.contains(std::to_string(recv_idx)) ? id_map[std::to_string(recv_idx)] : ""}}},
                     {"value", processed[emit_idx]}});

                tracer_throughput_total.push_back(
                    {{"labels",
                      {{"type", cmpt_type},
                       {"source_name",
                        id_map.contains(std::to_string(emit_idx)) ? id_map[std::to_string(emit_idx)] : ""},
                       {"dest_name",
                        id_map.contains(std::to_string(recv_idx)) ? id_map[std::to_string(recv_idx)] : ""}}},
                     {"value", processed[emit_idx] / total_elapsed_seconds}});
            }
        }
    }
}

void ThroughputTracer::emit(const std::size_t emit_id,
                            const std::size_t recv_id,
                            const time_pt_t& emit_ts,
                            const time_pt_t& recv_ts,
                            const time_unit_ns_t& elapsed)
{
    auto offset = emit_id * max_nodes() + emit_id;

    m_latencies[offset] += elapsed.count();
    m_emitted_by[emit_id] += 1;
}

std::string ThroughputTracer::name_id() const
{
    return tracer_name;
}

void ThroughputTracer::receive(const std::size_t emit_id,
                               const std::size_t recv_id,
                               const time_pt_t& emit_ts,
                               const time_pt_t& recv_ts,
                               const time_unit_ns_t& elapsed)

{
    auto offset = emit_id * max_nodes() + recv_id;

    m_latencies[offset] += elapsed.count();
    m_received_by[recv_id] += 1;
}

TraceAggregatorBase::TraceAggregatorBase() = default;

void TraceAggregatorBase::process_tracer_data(const std::vector<std::shared_ptr<TracerBase>>& tracers,
                                              double trace_time_elapsed_seconds,
                                              std::size_t segment_node_count,
                                              const std::map<std::size_t, std::string>& cmpt_id_to_name)
{
    std::lock_guard<std::mutex> lock(m_update_mutex);

    m_elapsed_seconds = trace_time_elapsed_seconds;
    m_node_count      = segment_node_count;
    m_tracer_count    = tracers.size();
    m_json_data       = {
        {
            "metadata",
            {{"elapsed_time", m_elapsed_seconds}, {"tracer_count", m_tracer_count}, {"node_count", m_node_count}},
        },
        {"aggregations", {}}};

    if (!cmpt_id_to_name.empty())
    {
        m_json_data["metadata"]["id_map"] = convert_to_string_key_map(cmpt_id_to_name);
    }

    this->aggregate(tracers, m_json_data);
}

const json& TraceAggregatorBase::to_json()
{
    return m_json_data;
}

}  // namespace mrc::benchmarking
