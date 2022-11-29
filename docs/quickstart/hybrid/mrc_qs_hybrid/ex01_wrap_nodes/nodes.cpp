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

#include "mrc_qs_hybrid/data_object.hpp"

#include <glog/logging.h>
#include <mrc/runnable/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/pybind11.h>
#include <pymrc/node.hpp>

#include <memory>
#include <sstream>
#include <utility>

namespace mrc::quickstart::hybrid::ex01_wrap_nodes {
namespace py = pybind11;

class MyDataObjectSource : public mrc::pymrc::PythonSource<std::shared_ptr<common::DataObject>>
{
  public:
    MyDataObjectSource(size_t count) : PythonSource(build()), m_count(count) {}

  private:
    subscriber_fn_t build()
    {
        return [this](rxcpp::subscriber<source_type_t> output) {
            for (size_t i = 0; i < m_count; ++i)
            {
                // Create a new object
                auto obj = std::make_shared<common::DataObject>(MRC_CONCAT_STR("Instance-" << i), i);

                output.on_next(std::move(obj));
            }

            output.on_completed();
        };
    }

    size_t m_count;
};

class MyDataObjectNode
  : public mrc::pymrc::PythonNode<std::shared_ptr<common::DataObject>, std::shared_ptr<common::DataObject>>
{
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<common::DataObject>, std::shared_ptr<common::DataObject>>;

  public:
    MyDataObjectNode() : PythonNode(base_t::op_factory_from_sub_fn(build())) {}

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

class MyDataObjectSink : public mrc::pymrc::PythonSink<std::shared_ptr<common::DataObject>>
{
  public:
    MyDataObjectSink() : PythonSink(build_on_next(), build_on_complete()) {}

  private:
    on_next_fn_t build_on_next()
    {
        return [this](sink_type_t x) {
            // Log the received value
            LOG(INFO) << mrc::runnable::Context::get_runtime_context().info() << " Got value: " << x->to_string();
        };
    }

    on_complete_fn_t build_on_complete()
    {
        return [this]() {
            // Log that we are completed
            LOG(INFO) << mrc::runnable::Context::get_runtime_context().info() << " Completed";
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
    mrc::pymrc::import(m, "mrc.core.node");

    py::class_<mrc::segment::Object<MyDataObjectSource>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<MyDataObjectSource>>>(
        m, "MyDataObjectSource", py::multiple_inheritance())
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name, size_t count) {
                 auto stage = parent.construct_object<MyDataObjectSource>(name, count);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("count"));

    py::class_<mrc::segment::Object<MyDataObjectNode>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<MyDataObjectNode>>>(
        m, "MyDataObjectNode", py::multiple_inheritance())
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<MyDataObjectNode>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<mrc::segment::Object<MyDataObjectSink>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<MyDataObjectSink>>>(
        m, "MyDataObjectSink", py::multiple_inheritance())
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<MyDataObjectSink>(name);

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
}  // namespace mrc::quickstart::hybrid::ex01_wrap_nodes
