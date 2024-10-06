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

#include "pymrc/forward.hpp"
#include "pymrc/node.hpp"
#include "pymrc/port_builders.hpp"
#include "pymrc/types.hpp"
#include "pymrc/utilities/acquire_gil.hpp"
#include "pymrc/utils.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_connector.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace mrc::pytests {

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;

using namespace py::literals;

struct Base
{
    virtual ~Base() = default;
};

struct DerivedA : public Base
{};

struct DerivedB : public Base
{};

class PythonTestNodeMixin
{
  protected:
    PythonTestNodeMixin(std::string name, pymrc::PyHolder counter_dict) :
      m_name(std::move(name)),
      m_counters(std::move(counter_dict))
    {}

    void init_counter(const std::string& counter_name)
    {
        pymrc::AcquireGIL gil;

        std::string key = MRC_CONCAT_STR(m_name << "." << counter_name);

        if (m_counters)
        {
            m_counters[key.c_str()] = 0;
        }
    }

    void increment_counter(const std::string& counter_name)
    {
        pymrc::AcquireGIL gil;

        std::string key = MRC_CONCAT_STR(m_name << "." << counter_name);

        if (m_counters)
        {
            m_counters[key.c_str()] = m_counters.attr("get")(key.c_str(), 0).cast<int>() + 1;
        }
    }

    std::string m_name;
    pymrc::PyHolder m_counters;  // Dict
};

#define GENERATE_NODE_TYPES(base_class, class_prefix)          \
    class class_prefix##Base : public base_class<Base>         \
    {                                                          \
        using base_class<Base>::base_class;                    \
    };                                                         \
    class class_prefix##DerivedA : public base_class<DerivedA> \
    {                                                          \
        using base_class<DerivedA>::base_class;                \
    };                                                         \
    class class_prefix##DerivedB : public base_class<DerivedB> \
    {                                                          \
      public:                                                  \
        using base_class<DerivedB>::base_class;                \
    };

template <typename T>
class TestSourceImpl : public PythonTestNodeMixin
{
  public:
    using source_t = std::shared_ptr<T>;

    TestSourceImpl(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      PythonTestNodeMixin(std::move(name), std::move(counter)),
      m_msg_count(msg_count)
    {
        this->init_counter("on_next");
        this->init_counter("on_error");
        this->init_counter("on_completed");
    }

  protected:
    auto build()
    {
        return rxcpp::observable<>::create<source_t>([this](rxcpp::subscriber<source_t>& output) {
            for (size_t i = 0; i < m_msg_count; ++i)
            {
                output.on_next(std::make_shared<T>());
                this->increment_counter("on_next");
            }

            this->increment_counter("on_completed");
            output.on_completed();
        });
    }

    size_t m_msg_count{5};
};

template <typename T>
class TestSource : public pymrc::PythonSource<std::shared_ptr<T>>, public TestSourceImpl<T>
{
  public:
    using base_t = pymrc::PythonSource<std::shared_ptr<T>>;

    TestSource(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      base_t(),
      TestSourceImpl<T>(std::move(name), std::move(counter), msg_count)
    {
        this->set_observable(this->build());
    }
};

template <typename T>
class TestSourceComponent : public pymrc::PythonSourceComponent<std::shared_ptr<T>>, public TestSourceImpl<T>
{
  public:
    using base_t = pymrc::PythonSourceComponent<std::shared_ptr<T>>;

    TestSourceComponent(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      base_t(build()),
      TestSourceImpl<T>(std::move(name), std::move(counter), msg_count)
    {}

  private:
    //   Hide the base build since there is no RxSourceComponent
    typename base_t::get_data_fn_t build()
    {
        return [this](std::shared_ptr<T>& output) {
            if (m_count++ < this->m_msg_count)
            {
                output = std::make_shared<T>();

                this->increment_counter("on_next");

                return channel::Status::success;
            }

            this->increment_counter("on_completed");

            return channel::Status::closed;
        };
    }

    size_t m_count{0};
};

template <typename T>
class TestNodeImpl : public PythonTestNodeMixin
{
  public:
    using sink_type_t   = std::shared_ptr<T>;
    using source_type_t = std::shared_ptr<T>;

    TestNodeImpl(std::string name, pymrc::PyHolder counter) : PythonTestNodeMixin(std::move(name), std::move(counter))
    {
        this->init_counter("on_next");
        this->init_counter("on_error");
        this->init_counter("on_completed");
    }

  protected:
    auto build_operator()
    {
        return [this](rxcpp::observable<sink_type_t> input) {
            return input | rxcpp::operators::tap([this](sink_type_t x) {
                       // Forward on
                       this->increment_counter("on_next");
                   }) |
                   rxcpp::operators::finally([this]() {
                       this->increment_counter("on_completed");
                   });
        };
    }
};

template <typename T>
class TestNode : public pymrc::PythonNode<std::shared_ptr<T>, std::shared_ptr<T>>, public TestNodeImpl<T>
{
    using base_t = pymrc::PythonNode<std::shared_ptr<T>, std::shared_ptr<T>>;

  public:
    TestNode(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      TestNodeImpl<T>(std::move(name), std::move(counter))
    {
        this->make_stream(this->build_operator());
    }
};

template <typename T>
class TestNodeComponent : public pymrc::PythonNodeComponent<std::shared_ptr<T>, std::shared_ptr<T>>,
                          public TestNodeImpl<T>
{
    using base_t = pymrc::PythonNodeComponent<std::shared_ptr<T>, std::shared_ptr<T>>;

  public:
    TestNodeComponent(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      TestNodeImpl<T>(std::move(name), std::move(counter))
    {
        this->make_stream(this->build_operator());
    }
};

template <typename T>
class TestSinkImpl : public PythonTestNodeMixin
{
  public:
    using sink_type_t = std::shared_ptr<T>;

    TestSinkImpl(std::string name, pymrc::PyHolder counter) : PythonTestNodeMixin(std::move(name), std::move(counter))
    {
        this->init_counter("on_next");
        this->init_counter("on_error");
        this->init_counter("on_completed");
    }

    rxcpp::observer<std::shared_ptr<T>> build()
    {
        return rxcpp::make_observer_dynamic<sink_type_t>(
            [this](sink_type_t x) {
                this->increment_counter("on_next");
            },
            [this](std::exception_ptr ex) {
                this->increment_counter("on_error");
            },
            [this]() {
                this->increment_counter("on_completed");
            });
    }
};

template <typename T>
class TestSink : public pymrc::PythonSink<std::shared_ptr<T>>, public TestSinkImpl<T>
{
  public:
    TestSink(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      TestSinkImpl<T>(std::move(name), std::move(counter))
    {
        this->set_observer(this->build());
    }
};

template <typename T>
class TestSinkComponent : public pymrc::PythonSinkComponent<std::shared_ptr<T>>, public TestSinkImpl<T>
{
  public:
    TestSinkComponent(std::string name, pymrc::PyHolder counter, size_t msg_count = 5) :
      TestSinkImpl<T>(std::move(name), std::move(counter))
    {
        this->set_observer(this->build());
    }
};

GENERATE_NODE_TYPES(TestSource, Source);
GENERATE_NODE_TYPES(TestSourceComponent, SourceComponent);
GENERATE_NODE_TYPES(TestNode, Node);
GENERATE_NODE_TYPES(TestNodeComponent, NodeComponent);
GENERATE_NODE_TYPES(TestSink, Sink);
GENERATE_NODE_TYPES(TestSinkComponent, SinkComponent);

#define CREATE_TEST_NODE_CLASS(class_name)                                                                         \
    py::class_<segment::Object<class_name>,                                                                        \
               mrc::segment::ObjectProperties,                                                                     \
               std::shared_ptr<segment::Object<class_name>>>(py_mod, #class_name)                                  \
        .def(py::init<>(                                                                                           \
                 [](mrc::segment::IBuilder& parent, const std::string& name, py::dict counter, size_t msg_count) { \
                     auto stage = parent.construct_object<class_name>(name, name, std::move(counter), msg_count);  \
                     return stage;                                                                                 \
                 }),                                                                                               \
             py::arg("parent"),                                                                                    \
             py::arg("name"),                                                                                      \
             py::arg("counter"),                                                                                   \
             py::arg("msg_count") = 5);

PYBIND11_MODULE(test_edges_cpp, py_mod)
{
    py_mod.doc() = R"pbdoc()pbdoc";

    pymrc::import(py_mod, "mrc");
    pymrc::import(py_mod, "mrc.core.segment");

    py::class_<Base, std::shared_ptr<Base>>(py_mod, "Base").def(py::init<>([]() {
        return std::make_shared<Base>();
    }));
    mrc::pymrc::PortBuilderUtil::register_port_util<Base>();

    py::class_<DerivedA, Base, std::shared_ptr<DerivedA>>(py_mod, "DerivedA").def(py::init<>([]() {
        return std::make_shared<DerivedA>();
    }));
    mrc::pymrc::PortBuilderUtil::register_port_util<DerivedA>();

    py::class_<DerivedB, Base, std::shared_ptr<DerivedB>>(py_mod, "DerivedB").def(py::init<>([]() {
        return std::make_shared<DerivedB>();
    }));
    mrc::pymrc::PortBuilderUtil::register_port_util<DerivedB>();

    mrc::edge::EdgeConnector<py::object, pymrc::PyObjectHolder>::register_converter();
    mrc::edge::EdgeConnector<pymrc::PyObjectHolder, py::object>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<DerivedA>, std::shared_ptr<Base>>::register_converter();
    mrc::edge::EdgeConnector<std::shared_ptr<DerivedB>, std::shared_ptr<Base>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<Base>, std::shared_ptr<DerivedA>>::register_dynamic_cast_converter();
    mrc::edge::EdgeConnector<std::shared_ptr<Base>, std::shared_ptr<DerivedB>>::register_dynamic_cast_converter();

    CREATE_TEST_NODE_CLASS(SourceBase);
    CREATE_TEST_NODE_CLASS(SourceDerivedA);
    CREATE_TEST_NODE_CLASS(SourceDerivedB);

    CREATE_TEST_NODE_CLASS(NodeBase);
    CREATE_TEST_NODE_CLASS(NodeDerivedA);
    CREATE_TEST_NODE_CLASS(NodeDerivedB);

    CREATE_TEST_NODE_CLASS(SinkBase);
    CREATE_TEST_NODE_CLASS(SinkDerivedA);
    CREATE_TEST_NODE_CLASS(SinkDerivedB);

    CREATE_TEST_NODE_CLASS(SourceComponentBase);
    CREATE_TEST_NODE_CLASS(SourceComponentDerivedA);
    CREATE_TEST_NODE_CLASS(SourceComponentDerivedB);

    CREATE_TEST_NODE_CLASS(NodeComponentBase);
    CREATE_TEST_NODE_CLASS(NodeComponentDerivedA);
    CREATE_TEST_NODE_CLASS(NodeComponentDerivedB);

    CREATE_TEST_NODE_CLASS(SinkComponentBase);
    CREATE_TEST_NODE_CLASS(SinkComponentDerivedA);
    CREATE_TEST_NODE_CLASS(SinkComponentDerivedB);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pytests
