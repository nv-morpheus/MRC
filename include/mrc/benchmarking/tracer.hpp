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

#include "mrc/benchmarking/util.hpp"

#include <ext/new_allocator.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace mrc::benchmarking {
using namespace std::chrono_literals;

class TracerBase
{
  public:
    using time_pt_t      = std::chrono::steady_clock::time_point;
    using time_unit_ns_t = std::chrono::nanoseconds;

    virtual ~TracerBase() = default;
    TracerBase(std::size_t max_nodes);

    /**
     * @brief Get the maximum node count this tracer can hold state for.
     * @return max node count
     */
    std::size_t max_nodes() const;

    /**
     * @brief Human readable name for the type of tracer
     * @return Tracer name.
     */
    virtual std::string name_id() const = 0;

    /**
     * @brief Called when the tracer is emitted by a node into a channel. This update encapsulates information about
     *  the time between when this tracer was received by a node and is now being emitted. [recv] ---> node ops ---->
     * [emit]. In this particular case, the elapsed time indicates how long it took the tracer to traverse the node's
     * call chain.
     * @param emit_id Identifying index of the emitting node.
     * @param recv_id Identifying index of the receiving node.
     * @param emit_ts Timestamp when the emission occurred
     * @param recv_ts Timestamp indicating the last receive time for this tracer
     * @param elapsed Elapsed time (nsec) between when the tracer was last received and its emission time. (recv_ts -
     * emit_ts)
     */
    virtual void emit(std::size_t emit_id,
                      std::size_t recv_id,
                      const time_pt_t& emit_ts,
                      const time_pt_t& recv_ts,
                      const time_unit_ns_t& elapsed) = 0;

    /**
     * @brief Called when the tracer is received by a node from a channel. This update encapsulates information about
     * the time between when this tracer was emitted by a prior node, passed through a channel, and is now being
     * received. [emit] ---> time in channel ----> [recv]. In this particular case, the elapsed time indicates how long
     * it took the tracer to traverse the channel between the emit node and the recv node.
     * @param emit_id Identifying index of the emitting node.
     * @param recv_id Identifying index of the receiving node.
     * @param emit_ts Timestamp when the emission occurred
     * @param recv_ts Timestamp indicating the last receive time for this tracer
     * @param elapsed Elapsed time (nsec) between when the tracer was last emitted and its receive time. (emit_ts -
     * recv_ts)
     */
    virtual void receive(std::size_t emit_id,
                         std::size_t recv_id,
                         const time_pt_t& emit_ts,
                         const time_pt_t& recv_ts,
                         const time_unit_ns_t& elapsed) = 0;

  protected:
    const std::size_t m_max_nodes;
};

class LatencyTracer : public TracerBase
{
    static constexpr char tracer_name[] = "LatencyTracer";  // NOLINT

  public:
    ~LatencyTracer() override = default;
    LatencyTracer(std::size_t max_nodes);

    static void aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, nlohmann::json& j);

    void emit(std::size_t emit_id,
              std::size_t recv_id,
              const time_pt_t& emit_ts,
              const time_pt_t& recv_ts,
              const time_unit_ns_t& elapsed) override;

    std::string name_id() const override;

    void receive(std::size_t emit_id,
                 std::size_t recv_id,
                 const time_pt_t& emit_ts,
                 const time_pt_t& recv_ts,
                 const time_unit_ns_t& elapsed) override;

  private:
    std::vector<std::size_t> m_latencies;
};

struct ThroughputTracer : public TracerBase
{
    static constexpr char tracer_name[] = "ThroughputTracer";  // NOLINT

    ~ThroughputTracer() override = default;
    ThroughputTracer(std::size_t max_nodes);

    static void aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, nlohmann::json& j);

    void emit(std::size_t emit_id,
              std::size_t recv_id,
              const time_pt_t& emit_ts,
              const time_pt_t& recv_ts,
              const time_unit_ns_t& elapsed) override;

    std::string name_id() const override;

    void receive(std::size_t emit_id,
                 std::size_t recv_id,
                 const time_pt_t& emit_ts,
                 const time_pt_t& recv_ts,
                 const time_unit_ns_t& elapsed) override;

    std::vector<std::size_t> m_latencies;
    std::vector<std::size_t> m_received_by;
    std::vector<std::size_t> m_emitted_by;
};

template <typename DataTypeT, typename... TracerTypeT>
class TracerEnsemble : public TracerTypeT...
{
  public:
    using tuple_type_t   = TracerEnsemble<DataTypeT, TracerTypeT...>;
    using data_type_t    = DataTypeT;
    using tracer_types_t = typename std::tuple<TracerTypeT...>;

    static constexpr auto EnsembleSizeV = sizeof...(TracerTypeT);

    template <std::size_t N>
    using nth_t = typename std::tuple_element<N, tracer_types_t>::type;

    ~TracerEnsemble() = default;
    TracerEnsemble(std::size_t max_nodes) : m_max_nodes(max_nodes), TracerTypeT(max_nodes)... {};

    /**
     * @return Payload data for this tracer ensemble.
     */
    operator DataTypeT&()
    {
        return m_payload;
    }

    /**
     * @brief Assignment overload.
     * @param other Payload data to assign to this ensemble.
     * @return Reference to this ensemble.
     */
    DataTypeT& operator=(DataTypeT& other)
    {
        if (this == &other)
        {
            return *this;
        }

        m_payload = other;
        return *this;
    }

    /**
     * @breif Assignment overload.
     * @param other Payload data to assign to this ensemble.
     * @return Reference to this ensemble.
     */
    DataTypeT& operator=(DataTypeT&& other)
    {
        m_payload = std::move(other);
        return *this;
    }

    /**
     * @brief Emission handler for this ensemble. Calculates common information and propagates it to all base tracer
     * types.
     * Note: What is going on with the emission timestamps -- we're often interested in measuring time scales
     * between receive and emit time points that are at the nanosecond scale; while our steady_clock::now() call can
     * provide us with nanosecond resolution, it also takes some non-zero amount of time to return; Based on testing
     * this delay can be up to a microsecond, call this DeltaNow(), and let T0_ret be the actual time the now() call
     * completes.
     * T0 = steady_clock::now(); // T0 = T0_ret - DeltaNow()
     * | ---- DeltaNow() ---- | -- DeltaT -- | ---- DeltaNow() ---- |
     * T0                   T0_ret           T1                   T1_ret
     * T1 = steady_clock::now(); // T1 = T0 + DeltaNow() + DeltaT = T1_ret - DeltaNow()
     * T1 - T0 == (DeltaNow() + DeltaT)
     *
     * What we want is for tracer elapsed calculations to be independent (internally) of DeltaNow(), which is
     * accomplished as follows:
     *
     * Let DeltaNowEstimate == mean(n steady_clock::now() samples).
     * 1) Tracer is created:
     *   // Set m_recv_ts to be approximately equal to the time we expect steady_clock::now() to return.
     *   T0 = steady_clock()::now()
     *   T0_ret = T0 + DeltaNow()
     *   m_recv_ts = ~T0_ret = T0 + DeltaNowEstimate();
     * 2) Tracer is emitted:
     *   T1 = steady_clock::now()
     *   T1_ret = T1 + DeltaNow()
     *   m_emit_ts = T1 = T0_ret + DeltaT.
     *   // Compute m_recv_emit_elapsed, attempting to compensate for DeltaNow(). If for some reason DeltaNow() <
     *   // DeltaNowEstimate() for this call (maybe we're timing no-op nodes), clamp to 1 ns.
     *   m_recv_emit_elapsed = (m_emit_ts - m_recv_ts)
     *      = (T0 + DeltaNow() + DeltaT) - (T0 + DeltaNowEstimate()) = max(1, (DeltaNow() - DeltaNowEstimate()) +
     * DeltaT)
     *   // Update m_emit_ts to a new compensated time point that ignores work done to update our tracer
     *   m_emit_ts = steady_clock::now() + DeltaNowEstimate();
     *
     * @param nid Node ID where the emission occurred.
     */
    void emit(std::size_t nid)
    {
        m_emit_hop_id       = nid;
        m_emit_ts           = TimeUtil::get_current_time_point();
        m_recv_emit_elapsed = m_emit_ts > m_recv_ts ? TimeUtil::time_resolution_unit_t(m_emit_ts - m_recv_ts)
                                                    : TimeUtil::s_minimum_resolution;
        (TracerTypeT::emit(m_emit_hop_id, m_recv_hop_id, m_emit_ts, m_recv_ts, m_recv_emit_elapsed), ...);
        m_emit_ts = TimeUtil::get_delay_compensated_time_point();
    }

    /**
     * @brief Receive handler for this ensemble. Calculates common information and propagates it to all base tracer
     * types.
     * Note: see 'emit' for an explanation of time point computations.
     * @param nid Node ID where the emission occurred.
     */
    void receive(std::size_t nid)
    {
        m_recv_hop_id       = nid;
        m_recv_ts           = TimeUtil::get_current_time_point();
        m_emit_recv_elapsed = m_recv_ts > m_emit_ts ? TimeUtil::time_resolution_unit_t(m_recv_ts - m_emit_ts)
                                                    : TimeUtil::s_minimum_resolution;
        (TracerTypeT::receive(m_emit_hop_id, m_recv_hop_id, m_emit_ts, m_recv_ts, m_emit_recv_elapsed), ...);
        m_recv_ts = TimeUtil::get_delay_compensated_time_point();
    }

    /**
     * @brief Manually set the current receive hop ID. Helper function to help make tracer sources a bit more precise.
     * @param nid
     */
    void recv_hop_id(std::size_t nid)
    {
        CHECK(nid <= m_max_nodes);
        m_recv_hop_id = nid;
    }

    /**
     * @brief Reset timestamps for this trace ensemble.
     */
    void reset()
    {
        m_recv_ts = TimeUtil::get_delay_compensated_time_point();
        m_emit_ts = m_recv_ts;
    }

  protected:
    DataTypeT m_payload;

    TimeUtil::time_pt_t m_emit_ts{};
    TimeUtil::time_pt_t m_recv_ts{};
    TimeUtil::time_resolution_unit_t m_emit_recv_elapsed{0};
    TimeUtil::time_resolution_unit_t m_recv_emit_elapsed{0};

    std::size_t m_emit_hop_id{0};
    std::size_t m_recv_hop_id{0};

    std::size_t m_max_nodes;
};

class TraceAggregatorBase
{
  public:
    virtual ~TraceAggregatorBase() = default;
    TraceAggregatorBase();

    /**
     * Given a set of tracers and some trace metadata, aggregate the information across tracers and tracer types
     * of the TracerEnsemble.
     *
     * @param tracers Vector of tracers to aggregate
     * @param trace_time_elapsed_seconds Total elapsed trace time
     * @param segment_node_count Total segment nodes in traced segment
     * @param cmpt_id_to_name Map of int id to std::string names
     */
    void process_tracer_data(const std::vector<std::shared_ptr<TracerBase>>& tracers,
                             double trace_time_elapsed_seconds,
                             std::size_t segment_node_count,
                             const std::map<std::size_t, std::string>& cmpt_id_to_name);

    /**
     * @brief Return aggregated JSON data.
     * @return JSON data.
     */
    const nlohmann::json& to_json();

  protected:
    /**
     * Virtual function is used to aggregate tracer ensembles on the derived class.
     * @param tracers tracers to aggregate.
     * @param j json object where aggregate data will be collected.
     */
    virtual void aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, nlohmann::json& j) = 0;

    /**
     * @brief Convert a map that uses numeric keys into one with stringified keys; this is just to address a deficiency
     * with the nlohmann library.
     * @tparam KeyTypeT Type of the numeric key
     * @tparam ValueTypeT Type of the mapped value
     * @param source_map Numeric key map to be converted.
     * @return dest_map, a map with stringified numeric keys.
     */
    template <typename KeyTypeT, typename ValueTypeT, typename = std::enable_if_t<std::is_arithmetic_v<KeyTypeT>>>
    std::map<std::string, ValueTypeT> convert_to_string_key_map(std::map<KeyTypeT, ValueTypeT> source_map)
    {
        std::map<std::string, ValueTypeT> dest_map;
        for (const auto& [key, val] : source_map)
        {
            dest_map[std::to_string(key)] = val;
        }

        return dest_map;
    }

    std::size_t m_tracer_count;
    std::size_t m_node_count;
    double m_elapsed_seconds;

    nlohmann::json m_json_data;
    std::mutex m_update_mutex;
};

/**
 * @brief Class used to aggregate a collection of tracer ensembles into a unified set of output metrics.
 * @tparam EnsembleTypeT Parameter indicating the collection of tracer types belonging to the ensemble.
 */
template <typename EnsembleTypeT>
class TraceAggregator : public TraceAggregatorBase
{
  public:
    ~TraceAggregator() override = default;
    TraceAggregator() : TraceAggregatorBase() {}

  protected:
    /**
     * @brief Proxy member function for calling the type aware static aggregation function.
     * @param tracers Collection of tracers to aggregate
     * @param j json object where collection data is aggregated.
     */
    void aggregate(std::vector<std::shared_ptr<TracerBase>> tracers, nlohmann::json& j) override
    {
        aggregate_components(tracers, j);
    }

    /**
     * @brief Aggregate data from a vector of tracers, across all ensemble tracers and populate 'j' with the results.
     * @tparam N Initial tracer type index to aggregate from. Used to iterate over all tracer types within an ensemble.
     * @param tracers vector of shared pointers to a collection of tracer ensembles.
     * @param j json object to populate with aggregate data.
     */
    template <std::size_t N = 0>
    static void aggregate_components(std::vector<std::shared_ptr<TracerBase>> tracers, nlohmann::json& j)
    {
        if constexpr (N < EnsembleTypeT::EnsembleSizeV)
        {
            EnsembleTypeT::template nth_t<N>::aggregate(tracers, j);
            aggregate_components<N + 1>(tracers, j);
        }
    }
};
}  // namespace mrc::benchmarking
