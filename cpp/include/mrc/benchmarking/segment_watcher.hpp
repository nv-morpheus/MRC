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

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/benchmarking/tracer.hpp"
#include "mrc/core/executor.hpp"

#include <boost/fiber/barrier.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::benchmarking {

/**
 * @brief
 * @tparam TracerTypeT The Tracer variant that the SegmentWatcher will use.
 */
template <typename TracerTypeT>
class SegmentWatcher
{
  public:
    using clock_t   = std::chrono::steady_clock;
    using time_pt_t = std::chrono::time_point<std::chrono::steady_clock>;

    ~SegmentWatcher() = default;
    SegmentWatcher(std::shared_ptr<Executor> executor);
    SegmentWatcher(std::shared_ptr<Executor> executor, std::function<void(TracerTypeT&)> payload_init);

    [[nodiscard]] bool tracing() const;

    std::size_t get_or_create_node_entry(const std::string& id);

    // These don't work for some reason. rxcpp's templating system doesn't like the returned lambdas.
    // Leaving it for now in case there is time to revisit at some point.
    decltype(auto) create_tracer_emit_tap(const std::string& id);

    decltype(auto) create_tracer_receive_tap(const std::string& id);

    decltype(auto) create_tracer_sink_lambda(const std::string& id, std::function<void(TracerTypeT&)> sink_f);

    template <bool ForceTracerSequencing = false>
    decltype(auto) create_rx_tracer_source(const std::string& id);

    /**
     * @brief Initialize the watcher, ensures that the segment is running and that the watcher state is
     * set to run.
     */
    void run();

    /**
     * @brief Starts the tracing process on the SegmentWatcher, if the segment has not yet been started call start on it
     * then set the active tracing flag and update the timestamp.
     * @param clear : clear existing state before starting the trace.
     */
    void start_trace(bool clear = false);

    /**
     * @brief Stop the tracing process. Wait on any in-flight tracers before returning, and set the tracing end time.
     */
    void stop_trace();

    /**
     * @brief Stop the segment watcher and its segment.
     */
    void shutdown();

    /**
     * @brief Resets global trace statistics, watcher process counts, and collected tracers.
     */
    void reset();

    /**
     * @brief Begins tracing on the watched segment and continues until signaled by m_cond_wait.
     */
    void trace_until_notified();

    /**
     * @brief Schedules a completion handler to be called once m_cond_wait is signaled.
     * @param completion_handler callback to run once tracing has finished.
     */
    void on_trace_complete(std::function<void()> completion_handler);

    /**
     * @brief Aggregates each of the tracers captured by the Watcher into a summarized form.
     * @return TraceAggregator object.
     */
    std::shared_ptr<TraceAggregatorBase> aggregate_tracers();

    /**
     * @brief Check run state of SegmentWatcher
     * @return true if running, false otherwise
     */
    bool is_running() const;

    /**
     * @brief Set the watcher's tracer count. This is the total number of tracers which will be emitted during tracing
     * cycle.
     * @param count Number of tracers
     */
    void tracer_count(std::size_t count);

    /**
     * @brief Check the current tracer count
     * @return tracer count
     */
    std::size_t tracer_count() const;

    /**
     * @brief Set's the function that will be used to initialize tracer payload data.
     * @param payload_init
     */
    void payload_initializer(std::function<void(TracerTypeT&)> payload_init);

  private:
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_tracing{false};
    std::atomic<bool> m_segment_started{false};
    std::atomic<bool> m_latency_cycle_ready{true};

    std::shared_ptr<Executor> m_executor;

    std::mutex m_mutex;
    boost::fibers::condition_variable_any m_cond_wake;
    boost::fibers::barrier m_sync_barrier{2};

    std::size_t m_node_count{0};
    std::size_t m_max_nodes{0};
    std::size_t m_count{0};
    std::size_t m_count_max{1};

    time_pt_t m_tracing_start_ns;
    time_pt_t m_tracing_end_ns;

    std::function<void(TracerTypeT&)> m_payload_init{nullptr};

    std::map<std::string, std::size_t> m_nodeid;
    std::map<std::size_t, std::string> m_id2name;

    std::vector<std::shared_ptr<TracerBase>> m_tracers;

    void shutdown_watcher();
    void shutdown_segment();
};

template <typename TracerTypeT>
decltype(auto) SegmentWatcher<TracerTypeT>::create_tracer_emit_tap(const std::string& id)
{
    auto idx = get_or_create_node_entry(id);

    return rxcpp::operators::tap([idx](std::shared_ptr<TracerTypeT> tracer) { tracer->emit(idx); });
}

template <typename TracerTypeT>
decltype(auto) SegmentWatcher<TracerTypeT>::create_tracer_receive_tap(const std::string& id)

{
    auto idx = get_or_create_node_entry(id);

    return rxcpp::operators::tap([idx](std::shared_ptr<TracerTypeT> tracer) { tracer->receive(idx); });
}

template <typename TracerTypeT>
decltype(auto) SegmentWatcher<TracerTypeT>::create_tracer_sink_lambda(const std::string& id,
                                                                      std::function<void(TracerTypeT&)> sink_f)
{
    auto idx = get_or_create_node_entry(id);

    auto tracer_lambda = [this, sink_f, idx](std::shared_ptr<TracerTypeT> tracer) {
        tracer->receive(idx);
        sink_f(*tracer);
        tracer->emit(idx);

        m_latency_cycle_ready = true;
        m_tracers.push_back(tracer);
        m_count++;
    };

    return tracer_lambda;
}

template <typename TracerTypeT>
template <bool ForceTracerSequencing>
decltype(auto) SegmentWatcher<TracerTypeT>::create_rx_tracer_source(const std::string& id)
{
    auto idx = get_or_create_node_entry(id);

    auto tracer_source = [this, idx](rxcpp::subscriber<std::shared_ptr<TracerTypeT>> s) {
        while (is_running())
        {
            auto tid     = 0;
            auto emitted = 0;

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                // If we're not currently tracing, then we need to wait for the signal to start.
                if (!tracing())
                {
                    VLOG(5) << "Entering test cycle, sleeping until notified." << std::endl;

                    m_cond_wake.wait(lock);
                }

                // Make sure we're still running, if not, exit.
                if (!is_running())
                {
                    VLOG(5) << "Shutdown initiated while waiting for test cycle. Exiting." << std::endl;
                    break;
                }
            }

            VLOG(5) << "Waking and starting test cycle." << std::endl;
            m_latency_cycle_ready = true;
            while (m_count < m_count_max)
            {
                if constexpr (ForceTracerSequencing)
                {
                    while (!m_latency_cycle_ready && is_running())
                    {
                        // The current tracer object is still in flight. Wait till it's done.
                        // TODO (Devin): We don't have to use fibers. Should update this to reflect whatever
                        //  the runnable context is.
                        boost::this_fiber::yield();
                    }
                    m_latency_cycle_ready = false;
                }

                if (emitted < m_count_max)
                {
                    auto sp = std::make_shared<TracerTypeT>(m_max_nodes);
                    if (m_payload_init)
                    {
                        m_payload_init(*sp.get());
                    }
                    sp->recv_hop_id(idx);
                    sp->reset();
                    sp->emit(idx);
                    s.on_next(sp);
                    ++emitted;
                }
                else
                {
                    // We've emitted all the tracers we're going to, but they haven't all been
                    // processed.
                    boost::this_fiber::yield();
                }
            }
            VLOG(5) << "Test cycle complete, notifying sleepers" << std::endl;

            // Notify watchers that this test cycle has finished.
            m_cond_wake.notify_all();

            // Wait until watchers have synchronized and are ready for us to continue
            m_sync_barrier.wait();
        }

        VLOG(5) << "TracerEnsemble source is exiting." << std::endl;
        s.on_completed();
    };

    return tracer_source;
}

template <typename TracerTypeT>
SegmentWatcher<TracerTypeT>::SegmentWatcher(std::shared_ptr<Executor> executor) : m_executor(std::move(executor))
{}

template <typename TracerTypeT>
SegmentWatcher<TracerTypeT>::SegmentWatcher(std::shared_ptr<Executor> executor,
                                            std::function<void(TracerTypeT&)> payload_init) :
  m_executor(std::move(executor)),
  m_payload_init(payload_init)
{}

template <typename TracerTypeT>
[[nodiscard]] bool SegmentWatcher<TracerTypeT>::tracing() const
{
    return m_tracing;
}

template <typename TracerTypeT>
std::size_t SegmentWatcher<TracerTypeT>::get_or_create_node_entry(const std::string& id)
{
    auto node = m_nodeid.find(id);
    if (node != m_nodeid.end())
    {
        return node->second;
    }

    auto n      = m_node_count++;
    m_max_nodes = m_node_count;

    m_nodeid[id] = n;
    m_id2name[n] = id;

    return n;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::run()
{
    VLOG(5) << "Run called" << std::endl;
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_running)
    {
        return;
    }

    m_running = true;
    if (!m_segment_started)
    {
        m_executor->start();
        m_segment_started = true;
    }
    VLOG(5) << "Run completed" << std::endl;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::start_trace(bool clear)
{
    VLOG(5) << "Start_trace called" << std::endl;
    if (!m_running)
    {
        run();
    }

    if (m_tracing)
    {
        return;
    }

    if (clear)
    {
        reset();
    }

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_tracing          = true;
        m_tracing_start_ns = std::chrono::steady_clock::now();
    }

    // Notify any tracers to wake up
    m_cond_wake.notify_all();
    VLOG(5) << "Start_trace completed" << std::endl;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::stop_trace()
{
    bool t{true};
    VLOG(5) << "stop called" << std::endl;

    if (!std::atomic_compare_exchange_strong(&m_tracing, &t, false))
    {
        return;
    }

    m_tracing        = false;
    m_tracing_end_ns = std::chrono::steady_clock::now();

    // Notify all tracers and wait for them to sync
    m_cond_wake.notify_all();
    m_sync_barrier.wait();
    VLOG(5) << "stop completed" << std::endl;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::shutdown()
{
    VLOG(5) << "Shutdown called." << std::endl << std::flush;
    stop_trace();
    shutdown_watcher();
    shutdown_segment();
    VLOG(5) << "Shutdown completed." << std::endl << std::flush;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::reset()
{
    VLOG(5) << "Reset called." << std::endl;
    std::unique_lock<std::mutex> lock(m_mutex);
    m_count = 0;
    m_tracers.clear();
    TraceStatistics::reset();
    VLOG(5) << "Reset complete." << std::endl;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::trace_until_notified()
{
    start_trace();
    on_trace_complete([this]() {
        VLOG(5) << "Calling completion handler" << std::endl;
        stop_trace();
    });
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::on_trace_complete(std::function<void()> completion_handler)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        VLOG(5) << "Waiting on trace completed signal" << std::endl;
        m_cond_wake.wait(lock);
    }
    completion_handler();
}

template <typename TracerTypeT>
std::shared_ptr<TraceAggregatorBase> SegmentWatcher<TracerTypeT>::aggregate_tracers()
{
    using namespace nlohmann;
    auto elapsed    = m_tracing_end_ns - m_tracing_start_ns;
    auto aggregator = std::make_shared<TraceAggregator<TracerTypeT>>();
    aggregator->process_tracer_data(m_tracers, elapsed.count() / 1e9, m_max_nodes, m_id2name);

    return aggregator;
}

template <typename TracerTypeT>
bool SegmentWatcher<TracerTypeT>::is_running() const
{
    return m_running;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::tracer_count(std::size_t count)
{
    CHECK(count > 0);

    m_count_max = count;
    m_tracers.reserve(m_count_max);
}

template <typename TracerTypeT>
std::size_t SegmentWatcher<TracerTypeT>::tracer_count() const
{
    return m_tracers.size();
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::payload_initializer(std::function<void(TracerTypeT&)> payload_init)
{
    m_payload_init = payload_init;
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::shutdown_watcher()
{
    VLOG(5) << "Shutdown watcher called" << std::endl;

    m_running = false;
    m_cond_wake.notify_all();
}

template <typename TracerTypeT>
void SegmentWatcher<TracerTypeT>::shutdown_segment()
{
    VLOG(5) << "Calling notify all in segment shutdown" << std::endl;

    m_executor->join();
    m_executor.reset();
}

}  // namespace mrc::benchmarking
