/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/watchers.hpp"

#include "pymrc/executor.hpp"
#include "pymrc/pipeline.hpp"
#include "pymrc/utils.hpp"

#include "mrc/benchmarking/tracer.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace mrc::pymrc {

namespace py = pybind11;

LatencyWatcher::LatencyWatcher(std::shared_ptr<pymrc::Executor> executor) :
  latency_watcher_t(executor->get_executor()),
  m_executor(executor)
{
    auto payload_initializer = [](latency_ensemble_t& latency_ensemble) {
        py::gil_scoped_acquire gil;
        latency_ensemble = py::none();
    };

    this->payload_initializer(payload_initializer);
}

LatencyWatcher::LatencyWatcher(std::shared_ptr<pymrc::Executor> executor,
                               std::function<void(latency_ensemble_t&)> payload_init) :
  latency_watcher_t(executor->get_executor(), payload_init),
  m_executor(executor)
{}

void LatencyWatcher::make_segment(const std::string& name,
                                  const std::function<void(mrc::segment::IBuilder&, LatencyWatcher&)>& init)
{
    pymrc::Pipeline pipeline;

    auto tracer_init_wrapper = [this, init](mrc::segment::IBuilder& seg) {
        init(seg, *this);
    };

    pipeline.make_segment(name, tracer_init_wrapper);
    m_executor->register_pipeline(pipeline);
}

std::shared_ptr<mrc::segment::ObjectProperties> LatencyWatcher::make_tracer_source(mrc::segment::IBuilder& seg,
                                                                                   const std::string& name,
                                                                                   bool force_sequential)
{
    using data_type_t = std::shared_ptr<pymrc::latency_ensemble_t>;

    if (force_sequential)
    {
        return seg.make_source<data_type_t>(name, this->create_rx_tracer_source<true>(name));
    }

    return seg.make_source<data_type_t>(name, this->create_rx_tracer_source<false>(name));
}

std::shared_ptr<mrc::segment::ObjectProperties> LatencyWatcher::make_traced_node(
    mrc::segment::IBuilder& seg,
    const std::string& name,
    std::function<py::object(py::object py_obj)> map_f)
{
    using data_type_t = std::shared_ptr<pymrc::latency_ensemble_t>;
    auto internal_idx = this->get_or_create_node_entry(name);

    auto trace_node = seg.make_node<data_type_t, data_type_t>(name,
                                                              rxcpp::operators::tap([internal_idx](data_type_t tracer) {
                                                                  tracer->receive(internal_idx);
                                                              }),
                                                              rxcpp::operators::map([map_f](data_type_t tracer) {
                                                                  py::gil_scoped_acquire gil;
                                                                  *tracer = map_f(*tracer);  // Implicit operator
                                                                                             // conversion, operates on
                                                                                             // tracer's

                                                                  return tracer;
                                                              }),
                                                              rxcpp::operators::tap([internal_idx](data_type_t tracer) {
                                                                  tracer->emit(internal_idx);
                                                              }));

    return trace_node;  // std::static_pointer_cast<Node>(trace_node);
}

std::shared_ptr<mrc::segment::ObjectProperties> LatencyWatcher::make_tracer_sink(
    mrc::segment::IBuilder& seg,
    const std::string& name,
    std::function<void(py::object py_obj)> sink_f)
{
    using data_type_t    = std::shared_ptr<pymrc::latency_ensemble_t>;
    auto compound_sink_f = [sink_f](pymrc::latency_ensemble_t& data) {
        py::gil_scoped_acquire gil;
        sink_f(data);
    };
    auto tracer_sink_f = this->create_tracer_sink_lambda(name, compound_sink_f);
    auto tracer_sink   = seg.make_sink<data_type_t>(name, tracer_sink_f);

    return tracer_sink;
}

py::dict LatencyWatcher::aggregate_tracers_as_pydict()
{
    nlohmann::json j = latency_watcher_t::aggregate_tracers()->to_json();

    return pymrc::cast_from_json(j);
}

ThroughputWatcher::ThroughputWatcher(std::shared_ptr<pymrc::Executor> executor) :
  throughput_watcher_t(executor->get_executor()),
  m_executor(executor)
{
    using throughput_ensemble_t =
        mrc::benchmarking::TracerEnsemble<pybind11::object, mrc::benchmarking::ThroughputTracer>;

    auto payload_initializer = [](throughput_ensemble_t& le) {
        py::gil_scoped_acquire gil;
        le = py::none();
    };

    this->payload_initializer(payload_initializer);
}

ThroughputWatcher::ThroughputWatcher(std::shared_ptr<pymrc::Executor> executor,
                                     std::function<void(throughput_ensemble_t&)> payload_init) :
  throughput_watcher_t(executor->get_executor(), payload_init),
  m_executor(executor)
{}

void ThroughputWatcher::make_segment(const std::string& name,
                                     const std::function<void(mrc::segment::IBuilder&, ThroughputWatcher&)>& init)
{
    pymrc::Pipeline pipeline;

    auto tracer_init_wrapper = [this, init](mrc::segment::IBuilder& seg) {
        init(seg, *this);
    };

    pipeline.make_segment(name, tracer_init_wrapper);
    m_executor->register_pipeline(pipeline);
}

std::shared_ptr<mrc::segment::ObjectProperties> ThroughputWatcher::make_tracer_source(mrc::segment::IBuilder& seg,
                                                                                      const std::string& name,
                                                                                      bool force_sequential)
{
    using data_type_t = std::shared_ptr<pymrc::throughput_ensemble_t>;

    if (force_sequential)
    {
        return seg.make_source<data_type_t>(name, this->create_rx_tracer_source<true>(name));
    }

    return seg.make_source<data_type_t>(name, this->create_rx_tracer_source<false>(name));
}

std::shared_ptr<mrc::segment::ObjectProperties> ThroughputWatcher::make_traced_node(
    mrc::segment::IBuilder& seg,
    const std::string& name,
    std::function<py::object(py::object x)> map_f)
{
    using data_type_t = std::shared_ptr<pymrc::throughput_ensemble_t>;
    auto internal_idx = this->get_or_create_node_entry(name);

    auto trace_node = seg.make_node<data_type_t, data_type_t>(name,
                                                              rxcpp::operators::tap([internal_idx](data_type_t tracer) {
                                                                  tracer->receive(internal_idx);
                                                              }),
                                                              rxcpp::operators::map([map_f](data_type_t tracer) {
                                                                  py::gil_scoped_acquire gil;
                                                                  *tracer = map_f(*tracer);  // Implicit operator
                                                                                             // conversion, operates on
                                                                                             // tracer's

                                                                  return tracer;
                                                              }),
                                                              rxcpp::operators::tap([internal_idx](data_type_t tracer) {
                                                                  tracer->emit(internal_idx);
                                                              }));

    return trace_node;
}

std::shared_ptr<mrc::segment::ObjectProperties> ThroughputWatcher::make_tracer_sink(
    mrc::segment::IBuilder& seg,
    const std::string& name,
    std::function<void(py::object x)> sink_f)
{
    using data_type_t    = std::shared_ptr<pymrc::throughput_ensemble_t>;
    auto compound_sink_f = [sink_f](pymrc::throughput_ensemble_t& data) {
        py::gil_scoped_acquire gil;
        sink_f(data);
    };
    auto tracer_sink_f = this->create_tracer_sink_lambda(name, compound_sink_f);
    auto tracer_sink   = seg.make_sink<data_type_t>(name, tracer_sink_f);

    return tracer_sink;
}

py::dict ThroughputWatcher::aggregate_tracers_as_pydict()
{
    nlohmann::json j = throughput_watcher_t::aggregate_tracers()->to_json();

    return pymrc::cast_from_json(j);
}

}  // namespace mrc::pymrc
