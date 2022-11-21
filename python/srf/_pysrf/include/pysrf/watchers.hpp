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

#include "srf/benchmarking/segment_watcher.hpp"
#include "srf/benchmarking/tracer.hpp"

#include <pybind11/pytypes.h>  // for object, dict

#include <functional>
#include <memory>
#include <string>

namespace srf::segment {
class Builder;
struct ObjectProperties;
}  // namespace srf::segment

namespace srf::pysrf {
class Executor;
}

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

using latency_ensemble_t = srf::benchmarking::TracerEnsemble<pybind11::object, srf::benchmarking::LatencyTracer>;
using latency_watcher_t  = srf::benchmarking::SegmentWatcher<latency_ensemble_t>;

using throughput_ensemble_t = srf::benchmarking::TracerEnsemble<pybind11::object, srf::benchmarking::ThroughputTracer>;
using throughput_watcher_t  = srf::benchmarking::SegmentWatcher<throughput_ensemble_t>;

class LatencyWatcher : public latency_watcher_t
{
  public:
    LatencyWatcher(std::shared_ptr<pysrf::Executor> executor);
    LatencyWatcher(std::shared_ptr<pysrf::Executor> executor, std::function<void(latency_ensemble_t&)> payload_init);

    void make_segment(const std::string& name,
                      const std::function<void(srf::segment::Builder&, LatencyWatcher&)>& init);
    std::shared_ptr<srf::segment::ObjectProperties> make_tracer_source(srf::segment::Builder& seg,
                                                                       const std::string& name,
                                                                       bool force_sequential = false);
    std::shared_ptr<srf::segment::ObjectProperties> make_traced_node(
        srf::segment::Builder& seg,
        const std::string& name,
        std::function<pybind11::object(pybind11::object py_obj)> map_f);
    std::shared_ptr<srf::segment::ObjectProperties> make_tracer_sink(
        srf::segment::Builder& seg, const std::string& name, std::function<void(pybind11::object py_obj)> sink_f);

    pybind11::dict aggregate_tracers_as_pydict();

  private:
    std::shared_ptr<pysrf::Executor> m_executor;
};

class ThroughputWatcher : public throughput_watcher_t
{
  public:
    ThroughputWatcher(std::shared_ptr<pysrf::Executor> executor);
    ThroughputWatcher(std::shared_ptr<pysrf::Executor> executor,
                      std::function<void(throughput_ensemble_t&)> payload_init);

    void make_segment(const std::string& name,
                      const std::function<void(srf::segment::Builder&, ThroughputWatcher&)>& init);
    std::shared_ptr<srf::segment::ObjectProperties> make_tracer_source(srf::segment::Builder& seg,
                                                                       const std::string& name,
                                                                       bool force_sequential = false);
    std::shared_ptr<srf::segment::ObjectProperties> make_traced_node(
        srf::segment::Builder& seg,
        const std::string& name,
        std::function<pybind11::object(pybind11::object py_obj)> map_f);
    std::shared_ptr<srf::segment::ObjectProperties> make_tracer_sink(
        srf::segment::Builder& seg, const std::string& name, std::function<void(pybind11::object py_obj)> sink_f);

    pybind11::dict aggregate_tracers_as_pydict();

  private:
    std::shared_ptr<pysrf::Executor> m_executor;
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
