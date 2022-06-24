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

#include "pysrf/types.hpp"  // IWYU pragma: keep
#include "pysrf/utils.hpp"

#include "srf/channel/forward.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_connector.hpp"
#include "srf/node/edge_registry.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <functional>  // for function, ref
#include <memory>      // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast, allocator, make_shared
#include <ostream>     // for operator<<
#include <typeindex>   // for type_index
#include <utility>     // for move, forward

namespace srf {

namespace node {

template <typename SourceT>
struct Edge<SourceT,
            pysrf::PyHolder,
            std::enable_if_t<!pybind11::detail::is_pyobject<SourceT>::value &&
                                 !std::is_convertible_v<SourceT, pybind11::object>,
                             void>> : public EdgeBase<SourceT, pysrf::PyHolder>
{
    using base_t = EdgeBase<SourceT, pysrf::PyHolder>;
    using typename base_t::sink_t;
    using typename base_t::source_t;

    using EdgeBase<source_t, sink_t>::EdgeBase;

    // We need to hold the GIL here, because casting from c++ -> pybind11::object allocates memory with Py_Malloc.
    // Its also important to note that you do not want to hold the GIL when calling m_output->await_write, as
    // that can trigger a deadlock with another fiber reading from the end of the channel

    channel::Status await_write(source_t&& data) final
    {
        pysrf::PyHolder py_data;
        {
            pybind11::gil_scoped_acquire gil;
            py_data = pybind11::cast(std::move(data));
        }

        return this->ingress().await_write(std::move(py_data));
    };
};

template <typename SinkT>
struct Edge<
    pysrf::PyHolder,
    SinkT,
    std::enable_if_t<!pybind11::detail::is_pyobject<SinkT>::value && !std::is_convertible_v<pybind11::object, SinkT>,
                     void>> : public EdgeBase<pysrf::PyHolder, SinkT>
{
    using base_t = EdgeBase<pysrf::PyHolder, SinkT>;
    using typename base_t::sink_t;
    using typename base_t::source_t;

    using EdgeBase<source_t, sink_t>::EdgeBase;

    // We don't hold the GIL in any of the *_write because we are explciitly releasing object's pointer, and casting it
    // to a c++ data type.

    channel::Status await_write(source_t&& data) override
    {
        sink_t _data;
        {
            pybind11::gil_scoped_acquire gil;
            _data = pybind11::cast<sink_t>(pybind11::object(std::move(data)));
        }

        return this->ingress().await_write(std::move(_data));
    }

    static void register_converter()
    {
        EdgeConnector<source_t, sink_t>::register_converter();
    }
};

template <>
struct Edge<pysrf::PyObjectHolder, pybind11::object, void> : public EdgeBase<pysrf::PyObjectHolder, pybind11::object>
{
    using base_t = EdgeBase<pysrf::PyObjectHolder, pybind11::object>;
    using typename base_t::sink_t;
    using typename base_t::source_t;

    using EdgeBase<source_t, sink_t>::EdgeBase;

    // We need to hold the GIL here, because casting from c++ -> pybind11::object allocates memory with Py_Malloc.
    // Its also important to note that you do not want to hold the GIL when calling m_output->await_write, as
    // that can trigger a deadlock with another fiber reading from the end of the channel

    channel::Status await_write(source_t&& data) final
    {
        pysrf::AcquireGIL gil;

        pybind11::object py_data = std::move(data);

        gil.release();

        return this->ingress().await_write(std::move(py_data));
    };
};

template <>
struct Edge<pybind11::object, pysrf::PyObjectHolder, void> : public EdgeBase<pybind11::object, pysrf::PyObjectHolder>
{
    using base_t = EdgeBase<pybind11::object, pysrf::PyObjectHolder>;
    using typename base_t::sink_t;
    using typename base_t::source_t;

    using EdgeBase<source_t, sink_t>::EdgeBase;

    // We don't hold the GIL in any of the *_write because we are explciitly releasing object's pointer, and casting it
    // to a c++ data type.

    channel::Status await_write(source_t&& data) override
    {
        // No need for the GIL
        sink_t _data = pysrf::PyObjectHolder(std::move(data));

        return this->ingress().await_write(std::move(_data));
    }

    static void register_converter()
    {
        EdgeConnector<source_t, sink_t>::register_converter();
    }
};

}  // namespace node

namespace pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

namespace detail {

template <typename InputT>
class PythonSinkTypeErased : public node::SinkTypeErased
{
  private:
    using node::SinkTypeErased::ingress_handle;

    std::shared_ptr<channel::IngressHandle> ingress_for_source_type(std::type_index source_type) final
    {
        if (source_type == typeid(PyHolder))
        {
            // Check to see if we have a conversion in pybind11
            if (pybind11::detail::get_type_info(this->sink_type(true), false))
            {
                // Shortcut the check to the the registered converters
                auto edge = std::make_shared<node::Edge<PyHolder, InputT>>(
                    std::dynamic_pointer_cast<channel::Ingress<InputT>>(this->ingress_handle()));
                auto handle = std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(edge);
                CHECK(handle);
                return handle;
            }
        }

        return node::SinkTypeErased::ingress_for_source_type(source_type);
    }
};

template <typename OutputT>
class PythonSourceTypeErased : public node::SourceTypeErased
{
    std::shared_ptr<channel::IngressHandle> ingress_adaptor_for_sink(node::SinkTypeErased& sink) final
    {
        // First check if there was a defined converter
        if (node::EdgeRegistry::has_converter(this->source_type(), sink.sink_type()))
        {
            return node::SourceTypeErased::ingress_adaptor_for_sink(sink);
        }

        // Check here to see if we can short circuit if both of the types are the same
        if (this->source_type(false) == sink.sink_type(false))
        {
            // Register an edge identity converter
            node::IdentityEdgeConnector<OutputT>::register_converter();

            return node::SourceTypeErased::ingress_adaptor_for_sink(sink);
        }

        // By this point several things have happened:
        // 1. Simple shortcut for matching types has failed. SourceT != SinkT
        // 2. We do not have a registered converter
        // 3. Both of our nodes are registered python nodes, but their source and sink types may not be registered

        // We can come up with an edge if one of the following is true:
        // 1. The source is a pybind11::object and the sink is registered with pybind11
        // 2. The sink is a pybind11::object and the source is registered with pybind11
        // 3. Neither is a pybind11::object but both types are registered with pybind11 (worst case, C++ -> py -> C++)

        auto writer_type = this->source_type(true);
        auto reader_type = sink.sink_type(true);

        // Check registrations with pybind11
        auto* writer_typei = pybind11::detail::get_type_info(this->source_type(true), false);
        auto* reader_typei = pybind11::detail::get_type_info(sink.sink_type(true), false);

        // Check if the source is a pybind11::object
        if (writer_type == typeid(PyHolder) && reader_typei)
        {
            return sink_ingress_adaptor_for_source_type(sink, writer_type);
            // return sink.ingress_for_source_type(writer_type);
        }

        // Check if the sink is a py::object
        if (reader_type == typeid(PyHolder) && writer_typei)
        {
            // TODO(MDD): To avoid a compound edge here, register an edge converter between OutputT and py::object
            node::EdgeConnector<OutputT, PyHolder>::register_converter();

            // Build the edge with the holder type
            // return sink.ingress_for_source_type(this->source_type());
            return sink_ingress_adaptor_for_source_type(sink, this->source_type());
        }

        // Check if both have been registered with pybind 11
        if (writer_typei && reader_typei)
        {
            // TODO(MDD): Check python types to see if they are compatible

            // Py types can be converted but need a compound edge. Build that here
            // auto py_to_sink_edge = std::dynamic_pointer_cast<channel::Ingress<pybind11::object>>(
            //    sink.ingress_for_source_type(typeid(pybind11::object)));
            auto py_to_sink_edge = std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(
                sink_ingress_adaptor_for_source_type(sink, typeid(PyHolder)));

            auto source_to_py_edge = std::make_shared<node::Edge<OutputT, PyHolder>>(py_to_sink_edge);

            LOG(WARNING) << "WARNING: A slow edge connection between C++ nodes '" << this->source_type_name()
                         << "' and '" << sink.sink_type_name()
                         << "' has been detected. Performance between "
                            "these nodes can be improved by registering an EdgeConverter at compile time. Without "
                            "this, conversion "
                            "to an intermediate python type will be necessary (i.e. C++ -> Python -> C++).";

            return std::dynamic_pointer_cast<channel::IngressHandle>(source_to_py_edge);
        }

        // Otherwise return base which most likely will fail
        return node::SourceTypeErased::ingress_adaptor_for_sink(sink);
    }
};

}  // namespace detail

template <typename InputT>
class PythonSink : public node::RxSink<InputT>, public detail::PythonSinkTypeErased<InputT>
{
    using base_t = node::RxSink<InputT>;

  public:
    using typename base_t::observer_t;

    using node::RxSink<InputT>::RxSink;
};

template <typename InputT, typename OutputT>
class PythonNode : public node::RxNode<InputT, OutputT>,
                   public detail::PythonSinkTypeErased<InputT>,
                   public detail::PythonSourceTypeErased<OutputT>
{
    using base_t = node::RxNode<InputT, OutputT>;

  public:
    using typename base_t::stream_fn_t;
    using subscribe_fn_t = std::function<rxcpp::subscription(rxcpp::observable<InputT>, rxcpp::subscriber<OutputT>)>;

    using node::RxNode<InputT, OutputT>::RxNode;

  protected:
    static auto op_factory_from_sub_fn(subscribe_fn_t sub_fn)
    {
        return [=](rxcpp::observable<InputT> input) {
            // Convert from the `subscription(observable, subscriber)` signature into an operator factor function
            // `observable(observable)`
            return rxcpp::observable<>::create<OutputT>([=](rxcpp::subscriber<OutputT> output) {
                // Call the wrapped function
                sub_fn(input, output);
            });
        };
    }

  private:
    channel::Status no_channel(OutputT&& data) final
    {
        if constexpr (pybind11::detail::is_pyobject<OutputT>::value)
        {
            pybind11::gil_scoped_acquire gil;
            OutputT tmp = std::move(data);
        }
        else
        {
            OutputT tmp = std::move(data);
        }

        return channel::Status::success;
    }
};

template <typename OutputT>
class PythonSource : public node::RxSource<OutputT>, public detail::PythonSourceTypeErased<OutputT>
{
    using base_t = node::RxSource<OutputT>;

  public:
    using subscriber_fn_t = std::function<void(rxcpp::subscriber<OutputT>& sub)>;

    PythonSource(const subscriber_fn_t& f) :
      base_t(rxcpp::observable<>::create<OutputT>([f](rxcpp::subscriber<OutputT>& s) {
          // Call the wrapped subscriber function
          f(s);
      }))
    {}
};

class SegmentObjectProxy
{
    // add name
};

#pragma GCC visibility pop

}  // namespace pysrf
}  // namespace srf
