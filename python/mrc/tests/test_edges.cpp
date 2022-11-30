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

#include "pymrc/forward.hpp"
#include "pymrc/node.hpp"
#include "pymrc/port_builders.hpp"
#include "pymrc/utils.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/manifold/egress.hpp"
#include "mrc/node/edge_connector.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <boost/fiber/future/future.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

// IWYU thinks we need vector for PythonNode
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <boost/hana/if.hpp>
// IWYU pragma: no_include <boost/fiber/context.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/shared_state.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/task_base.hpp>
// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>
// IWYU pragma: no_include <pybind11/detail/type_caster_base.h>
// IWYU pragma: no_include "rx-includes.hpp"

namespace mrc::pytests {

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;

struct Base
{};

struct DerivedA : public Base
{};

struct DerivedB : public Base
{};

class SourceDerivedB : public pymrc::PythonSource<std::shared_ptr<DerivedB>>
{
  public:
    using base_t = pymrc::PythonSource<std::shared_ptr<DerivedB>>;
    using typename base_t::subscriber_fn_t;

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

class SourcePyHolder : public pymrc::PythonSource<pymrc::PyObjectHolder>
{
  public:
    using base_t = pymrc::PythonSource<pymrc::PyObjectHolder>;
    using typename base_t::subscriber_fn_t;

    SourcePyHolder() : PythonSource(build()) {}

  private:
    subscriber_fn_t build()
    {
        return [this](rxcpp::subscriber<pymrc::PyObjectHolder>& output) {
            for (size_t i = 0; i < 5; ++i)
            {
                output.on_next(py::int_(10 + i));
            }
        };
    }
};

class NodeBase : public pymrc::PythonNode<std::shared_ptr<Base>, std::shared_ptr<Base>>
{
  public:
    using base_t = pymrc::PythonNode<std::shared_ptr<Base>, std::shared_ptr<Base>>;
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

class NodePyHolder : public pymrc::PythonNode<pymrc::PyObjectHolder, pymrc::PyObjectHolder>
{
  public:
    using base_t = pymrc::PythonNode<pymrc::PyObjectHolder, pymrc::PyObjectHolder>;
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
                    pymrc::AcquireGIL gil;

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

class SinkBase : public pymrc::PythonSink<std::shared_ptr<Base>>
{
    using base_t = pymrc::PythonSink<std::shared_ptr<Base>>;

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

PYBIND11_MODULE(test_edges_cpp, module)
{
    module.doc() = R"pbdoc()pbdoc";

    pymrc::import(module, "mrc");

    py::class_<Base, std::shared_ptr<Base>>(module, "Base").def(py::init<>([]() { return std::make_shared<Base>(); }));
    mrc::pymrc::PortBuilderUtil::register_port_util<Base>();

    py::class_<DerivedA, Base, std::shared_ptr<DerivedA>>(module, "DerivedA").def(py::init<>([]() {
        return std::make_shared<DerivedA>();
    }));
    mrc::pymrc::PortBuilderUtil::register_port_util<DerivedA>();

    py::class_<DerivedB, Base, std::shared_ptr<DerivedB>>(module, "DerivedB").def(py::init<>([]() {
        return std::make_shared<DerivedB>();
    }));
    mrc::pymrc::PortBuilderUtil::register_port_util<DerivedB>();

    mrc::node::EdgeConnector<py::object, pymrc::PyObjectHolder>::register_converter();
    mrc::node::EdgeConnector<pymrc::PyObjectHolder, py::object>::register_converter();

    py::class_<segment::Object<SourceDerivedB>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<segment::Object<SourceDerivedB>>>(module, "SourceDerivedB")
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SourceDerivedB>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<SourcePyHolder>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<segment::Object<SourcePyHolder>>>(module, "SourcePyHolder")
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SourcePyHolder>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<NodeBase>, mrc::segment::ObjectProperties, std::shared_ptr<segment::Object<NodeBase>>>(
        module, "NodeBase")
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<NodeBase>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<NodePyHolder>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<segment::Object<NodePyHolder>>>(module, "NodePyHolder")
        .def(py::init<>([](mrc::segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<NodePyHolder>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<segment::Object<SinkBase>, segment::ObjectProperties, std::shared_ptr<segment::Object<SinkBase>>>(
        module, "SinkBase")
        .def(py::init<>([](segment::Builder& parent, const std::string& name) {
                 auto stage = parent.construct_object<SinkBase>(name);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    module.attr("__version__") =
        MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "." << mrc_VERSION_PATCH);
}
}  // namespace mrc::pytests
