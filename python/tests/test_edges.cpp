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

#include "pysrf/forward.hpp"
#include "pysrf/node.hpp"
#include "pysrf/utils.hpp"

#include "srf/channel/status.hpp"
#include "srf/node/edge_connector.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

// IWYU thinks we need vector for PythonNode
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <vector>

namespace srf::pytests {

namespace py    = pybind11;
namespace pysrf = srf::pysrf;

struct Base
{};

struct DerivedA : public Base
{};

struct DerivedB : public Base
{};

class SourceDerivedB : public pysrf::PythonSource<std::shared_ptr<DerivedB>>
{
  public:
    using base_t = pysrf::PythonSource<std::shared_ptr<DerivedB>>;
    using typename base_t::subscriber_fn_t;
    // using base_t::reader_type_t;
    // using base_t::writer_type_t;

    SourceDerivedB() : PythonSource(build()) {}

  private:
    subscriber_fn_t build()
    {
        return [this](rxcpp::subscriber<std::shared_ptr<DerivedB>>& output) {
            for (size_t i = 0; i < 5; ++i)
            {
                output.on_next(std::make_shared<DerivedB>());
            }
        };
    }
};

class SourcePyHolder : public pysrf::PythonSource<pysrf::PyObjectHolder>
{
  public:
    using base_t = pysrf::PythonSource<pysrf::PyObjectHolder>;
    using typename base_t::subscriber_fn_t;
    // using base_t::reader_type_t;
    // using base_t::writer_type_t;

    SourcePyHolder() : PythonSource(build()) {}

  private:
    subscriber_fn_t build()
    {
        return [this](rxcpp::subscriber<pysrf::PyObjectHolder>& output) {
            for (size_t i = 0; i < 5; ++i)
            {
                output.on_next(py::int_(10 + i));
            }
        };
    }
};

class NodeBase : public pysrf::PythonNode<std::shared_ptr<Base>, std::shared_ptr<Base>>
{
  public:
    using base_t = pysrf::PythonNode<std::shared_ptr<Base>, std::shared_ptr<Base>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    NodeBase() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}

  private:
    subscribe_fn_t build_operator()
    {
        return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
            return input.subscribe(rxcpp::make_observer<sink_type_t>(
                [this, &output](sink_type_t x) {
                    // Forward on
                    output.on_next(std::move(x));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }
};

class NodePyHolder : public pysrf::PythonNode<pysrf::PyObjectHolder, pysrf::PyObjectHolder>
{
  public:
    using base_t = pysrf::PythonNode<pysrf::PyObjectHolder, pysrf::PyObjectHolder>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    NodePyHolder() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}

  private:
    subscribe_fn_t build_operator()
    {
        return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
            return input.subscribe(rxcpp::make_observer<sink_type_t>(
                [this, &output](sink_type_t x) {
                    // Need to Grab the GIL to temp manipulate the value
                    pysrf::AcquireGIL gil;

                    py::int_ int_val = py::object(std::move(x));

                    int_val = int_val.cast<int>() + 1;

                    gil.release();

                    output.on_next(std::move(int_val));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }
};

class SinkBase : public pysrf::PythonSink<std::shared_ptr<Base>>
{
    using base_t = pysrf::PythonSink<std::shared_ptr<Base>>;

    static rxcpp::observer<std::shared_ptr<Base>> build()
    {
        return rxcpp::make_observer_dynamic<sink_type_t>(
            [](sink_type_t x) {

            },
            [](std::exception_ptr ex) {},
            []() {
                // Complete
            });
    }

  public:
    using typename base_t::observer_t;
    using typename base_t::sink_type_t;

    SinkBase() : PythonSink(build()) {}
};

PYBIND11_MODULE(test_edges_cpp, m)
{
    m.doc() = R"pbdoc()pbdoc";

    pysrf::import(m, "srf");

    py::class_<Base, std::shared_ptr<Base>>(m, "Base").def(py::init<>([]() { return std::make_shared<Base>(); }));

    py::class_<DerivedA, Base, std::shared_ptr<DerivedA>>(m, "DerivedA").def(py::init<>([]() {
        return std::make_shared<DerivedA>();
    }));

    py::class_<DerivedB, Base, std::shared_ptr<DerivedB>>(m, "DerivedB").def(py::init<>([]() {
        return std::make_shared<DerivedB>();
    }));

    srf::node::EdgeConnector<py::object, pysrf::PyObjectHolder>::register_converter();
    srf::node::EdgeConnector<pysrf::PyObjectHolder, py::object>::register_converter();

    py::class_<segment::Object<SourceDerivedB>,
               srf::segment::ObjectProperties,
               std::shared_ptr<segment::Object<SourceDerivedB>>>(m, "SourceDerivedB")
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SourceDerivedB>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<SourcePyHolder>,
               srf::segment::ObjectProperties,
               std::shared_ptr<segment::Object<SourcePyHolder>>>(m, "SourcePyHolder")
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SourcePyHolder>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<NodeBase>, srf::segment::ObjectProperties, std::shared_ptr<segment::Object<NodeBase>>>(
        m, "NodeBase")
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<NodeBase>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<NodePyHolder>,
               srf::segment::ObjectProperties,
               std::shared_ptr<segment::Object<NodePyHolder>>>(m, "NodePyHolder")
        .def(py::init<>([](srf::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<NodePyHolder>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<SinkBase>, segment::ObjectProperties, std::shared_ptr<segment::Object<SinkBase>>>(
        m, "SinkBase")
        .def(py::init<>([](segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SinkBase>(name);

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
}  // namespace srf::pytests
