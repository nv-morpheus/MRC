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

#include "mrc/benchmarking/segment_watcher.hpp"
#include "mrc/benchmarking/tracer.hpp"

#include <pybind11/pytypes.h>  // for object, dict

#include <functional>
#include <memory>
#include <string>

namespace mrc::segment {
class Builder;
struct ObjectProperties;
}  // namespace mrc::segment

namespace mrc::pymrc {
class Executor;
}

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

using latency_ensemble_t = mrc::benchmarking::TracerEnsemble<pybind11::object, mrc::benchmarking::LatencyTracer>;
using latency_watcher_t  = mrc::benchmarking::SegmentWatcher<latency_ensemble_t>;

using throughput_ensemble_t = mrc::benchmarking::TracerEnsemble<pybind11::object, mrc::benchmarking::ThroughputTracer>;
using throughput_watcher_t  = mrc::benchmarking::SegmentWatcher<throughput_ensemble_t>;

class LatencyWatcher : public latency_watcher_t
{
  public:
    LatencyWatcher(std::shared_ptr<pymrc::Executor> executor);
    LatencyWatcher(std::shared_ptr<pymrc::Executor> executor, std::function<void(latency_ensemble_t&)> payload_init);

    void make_segment(const std::string& name,
                      const std::function<void(mrc::segment::Builder&, LatencyWatcher&)>& init);
    std::shared_ptr<mrc::segment::ObjectProperties> make_tracer_source(mrc::segment::Builder& seg,
                                                                       const std::string& name,
                                                                       bool force_sequential = false);
    std::shared_ptr<mrc::segment::ObjectProperties> make_traced_node(
        mrc::segment::Builder& seg,
        const std::string& name,
        std::function<pybind11::object(pybind11::object py_obj)> map_f);
    std::shared_ptr<mrc::segment::ObjectProperties> make_tracer_sink(
        mrc::segment::Builder& seg, const std::string& name, std::function<void(pybind11::object py_obj)> sink_f);

    pybind11::dict aggregate_tracers_as_pydict();

  private:
    std::shared_ptr<pymrc::Executor> m_executor;
};

class ThroughputWatcher : public throughput_watcher_t
{
  public:
    ThroughputWatcher(std::shared_ptr<pymrc::Executor> executor);
    ThroughputWatcher(std::shared_ptr<pymrc::Executor> executor,
                      std::function<void(throughput_ensemble_t&)> payload_init);

    void make_segment(const std::string& name,
                      const std::function<void(mrc::segment::Builder&, ThroughputWatcher&)>& init);
    std::shared_ptr<mrc::segment::ObjectProperties> make_tracer_source(mrc::segment::Builder& seg,
                                                                       const std::string& name,
                                                                       bool force_sequential = false);
    std::shared_ptr<mrc::segment::ObjectProperties> make_traced_node(
        mrc::segment::Builder& seg,
        const std::string& name,
        std::function<pybind11::object(pybind11::object py_obj)> map_f);
    std::shared_ptr<mrc::segment::ObjectProperties> make_tracer_sink(
        mrc::segment::Builder& seg, const std::string& name, std::function<void(pybind11::object py_obj)> sink_f);

    pybind11::dict aggregate_tracers_as_pydict();

  private:
    std::shared_ptr<pymrc::Executor> m_executor;
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
