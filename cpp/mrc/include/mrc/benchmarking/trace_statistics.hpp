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
#include "mrc/core/watcher.hpp"

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstddef>  // for size_t
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

namespace mrc::benchmarking {

/**
 * @brief Class used to store statistics gathered from internal nodes via watcher interfaces. The class consists
 * of various static elements that allow for a consistent global view + aggregation of thread local storage elements
 * which keep track of statistics on a per thread basis.
 */
class TraceStatistics : public WatcherInterface
{
    // Thread local map to store a name-unique per-thread TraceStatistics struct.
    thread_local static std::map<std::string, std::shared_ptr<TraceStatistics>> TraceObjectMap;

    // Multi-map containing a shared pointer to all TraceStatistics structures across all thread_local storage.
    // Primarily used to aggregate statistics across threads without blocking or having to run per-thread routines.
    static std::multimap<std::string, std::shared_ptr<TraceStatistics>> TraceObjectMultimap;
    static std::recursive_mutex s_state_mutex;

    // Performance optimization that assigns an initial number of 'slots' for watcher entrypoint handling. Slot count
    // should be greater than or equal to the number of WatchableEvents
    static constexpr std::size_t MaxSlotsV = 10;

    static bool s_trace_operators;
    static bool s_trace_operators_set_manually;

    static bool s_trace_channels;
    static bool s_trace_channels_set_manually;

    static bool s_initialized;

    static void init();

  public:
    /**
     * @brief Aggregate statistics across all running stats aware elements.
     * @return Return the aggregated statistics in json format.
     */
    static nlohmann::json aggregate();

    /**
     * @brief (Threadsafe) Retrieve the thread local stats object associated with a given unique name or create a new
     * one if it does not exist. Its worth noting that the TraceStatistics object retrived is not explicitly thread
     * local -- rather, the map associating uniquely named TraceStatistics objects for each thread is thread-local.
     * @param name Name of the uniquely identified stats object.
     * @return Shared pointer to the thread local stats object.
     */
    static std::shared_ptr<TraceStatistics> get_or_create(const std::string& name);

    /**
     * @brief Retrieve a multi-map containing pointers to all thread local TraceStatistics objects. Can be used for
     * custom aggregation.
     * @return Multi-map where the multiplicity of each entry will be equal to the number of unique threads which
     * have called 'get_or_create'.
     */
    [[maybe_unused]] static const std::multimap<std::string, std::shared_ptr<TraceStatistics>>&
    get_thread_local_results_map();

    /**
     * @brief Reset existing statistics, and call sync_state.
     */
    static void reset();

    /**
     * @brief Re-evaluate the current runtime environment. This allow for run-time trace toggling, by adding or
     * removing environment variables.
     */
    static void sync_state();

    /**
     * Manually set operator tracing flag. This will override any environmental setting until 'reset' is called.
     * @param flag tracing flag
     * @param sync_immediate Whether or not to immediately synchronize the tracing environment.
     */
    static void trace_operators(bool flag, bool sync_immediate = true);

    /**
     * @brief return bool tuple indicating if channels are being traced and if they were set manually.
     * @return
     */
    [[maybe_unused]] static std::tuple<bool, bool> trace_operators();

    /**
     * Manually set channel tracing flag. This will override any environmental setting until 'reset' is called.
     * @param flag tracing flag
     * @param sync_immediate Whether or not to immediately synchronize the tracing environment.
     */
    static void trace_channels(bool flag, bool sync_immediate = true);

    /**
     * @brief return bool tuple indicating if channels are being traced and if they were set manually.
     * @return
     */
    [[maybe_unused]] static std::tuple<bool, bool> trace_channels();

    ~TraceStatistics() override             = default;
    TraceStatistics(const TraceStatistics&) = delete;
    TraceStatistics(TraceStatistics&&)      = delete;

    nlohmann::json to_json() const;

    /**
     * @brief Watcher interface override.
     */
    void on_entry(const WatchableEvent& e, const void* data) override;

    /**
     * @brief Watcher interface override.
     */
    void on_exit(const WatchableEvent& e, bool rc, const void* data) override;

  private:
    TraceStatistics(const std::string& name);

    /**
     * @brief Entry point for channel event.
     */
    void channel_read_end();

    /**
     * @brief Entry point for channel event.
     */
    void channel_read_start();

    /**
     * @brief Entry point for channel event.
     */
    void channel_write_end();

    /**
     * @brief Entry point for channel event.
     */
    void channel_write_start();

    /**
     * @brief clear all counters on this stats object.
     */
    void clear();

    /**
     * @brief Data emission handler -- called when an node emits a data element to its source output.
     */
    void emit();

    /**
     * @brief Initializes the lookup tables for WatchableEvent handlers. These will be set to no-ops if
     * the related tracing flags are false, or the handling implementation otherwise.
     */
    void initialize_lookup_tables();

    /**
     * @brief Data receive handler -- called when an node pulls data from it's sink a data element to its source output.
     */
    void receive();

    /**
     * Thread ID where this stats object is created.
     * @return thread id
     */
    std::thread::id parent_id() const;

    const std::string& m_name;
    const std::thread::id m_parent_id;

    TimeUtil::time_pt_t m_start_time;
    TimeUtil::time_pt_t m_internal_chain_start;
    TimeUtil::time_pt_t m_channel_read_start;
    TimeUtil::time_pt_t m_channel_write_start;

    std::size_t m_emission_count{0};
    std::size_t m_receive_count{0};
    std::size_t m_channel_sink_reads{0};
    std::size_t m_channel_source_writes{0};
    std::size_t m_total_internal_elapsed_ns{0};
    std::size_t m_total_ch_read_elapsed_ns{0};
    std::size_t m_total_ch_write_elapsed_ns{0};

    std::array<std::function<void(void)>, MaxSlotsV> m_entry_lookup_table;
    std::array<std::function<void(void)>, MaxSlotsV> m_exit_lookup_table;
};

}  // namespace mrc::benchmarking
