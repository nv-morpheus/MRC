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

#include <srf_qs_hybrid/data_object.hpp>

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include "pysrf/node.hpp"
#include "srf/runnable/context.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"
#include "srf/utils/string_utils.hpp"

#include <memory>
#include <sstream>
#include <utility>

namespace srf::quickstart::hybrid::common {
namespace py = pybind11;

class DataObjectSource : public srf::pysrf::PythonSource<std::shared_ptr<common::DataObject>>
{
    // using base_t = srf::pysrf::PythonSource<std::shared_ptr<common::DataObject>>;
    // using typename base_t::source_type_t;
    // using typename base_t::subscriber_fn_t;

  public:
    DataObjectSource(size_t count) : PythonSource(build()), m_count(count) {}

  private:
    subscriber_fn_t build()
    {
        return [this](rxcpp::subscriber<source_type_t> output) {
            for (size_t i = 0; i < m_count; ++i)
            {
                // Create a new object
                auto obj = std::make_shared<common::DataObject>(SRF_CONCAT_STR("Instance-" << i), i);

                output.on_next(std::move(obj));
            }

            output.on_completed();
        };
    }

    size_t m_count;
};

class DataObjectNode
  : public srf::pysrf::PythonNode<std::shared_ptr<common::DataObject>, std::shared_ptr<common::DataObject>>
{
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<common::DataObject>, std::shared_ptr<common::DataObject>>;

  public:
    DataObjectNode() : PythonNode(base_t::op_factory_from_sub_fn(build())) {}

  private:
    subscribe_fn_t build()
    {
        return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
            return input.subscribe(rxcpp::make_observer<sink_type_t>(
                [this, &output](sink_type_t x) {
                    x->value *= 2;

                    output.on_next(std::move(x));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    size_t m_count;
};

class DataObjectSink : public srf::pysrf::PythonSink<std::shared_ptr<common::DataObject>>
{
    // using base_t = srf::pysrf::PythonSink<std::shared_ptr<common::DataObject>>;
    // using typename base_t::sink_type_t;

  public:
    DataObjectSink() : PythonSink(build_on_next(), build_on_complete()) {}

  private:
    on_next_fn_t build_on_next()
    {
        return [this](sink_type_t x) {
            // Log the received value
            LOG(INFO) << srf::runnable::Context::get_runtime_context().info() << " Got value: " << x->to_string();
        };
    }

    on_complete_fn_t build_on_complete()
    {
        return [this]() {
            // Log that we are completed
            LOG(INFO) << srf::runnable::Context::get_runtime_context().info() << " Completed";
        };
    }
};

PYBIND11_MODULE(nodes, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: quickstart
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    // Required for SegmentObject
    srf::pysrf::import(m, "srf.core.node");

    py::class_<srf::segment::Object<DataObjectSource>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<DataObjectSource>>>(
        m, "DataObjectSource", py::multiple_inheritance())
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name, size_t count) {
                 auto stage = parent.construct_object<DataObjectSource>(name, count);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("count"));

    py::class_<srf::segment::Object<DataObjectNode>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<DataObjectNode>>>(m, "DataObjectNode", py::multiple_inheritance())
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<DataObjectNode>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<srf::segment::Object<DataObjectSink>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<DataObjectSink>>>(m, "DataObjectSink", py::multiple_inheritance())
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<DataObjectSink>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::quickstart::hybrid::common
