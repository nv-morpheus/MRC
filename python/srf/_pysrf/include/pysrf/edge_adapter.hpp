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
#include "srf/node/edge_adapter_registry.hpp"
#include "srf/node/edge_builder.hpp"
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
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <functional>  // for function, ref
#include <memory>      // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast, allocator, make_shared
#include <ostream>     // for operator<<
#include <type_traits>
#include <typeindex>  // for type_index
#include <utility>    // for move, forward:

namespace srf::pysrf {

/**
 * @brief Utility struct which supports building pySRF source/sink adapter functions which can be registered with
 * the EdgeAdapterRegistry.
 */
struct EdgeAdapterUtil
{
    using source_adapter_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(
        srf::node::SourcePropertiesBase&, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle>)>;
    using sink_adapter_fn_t   = std::function<std::shared_ptr<channel::IngressHandle>(
        std::type_index, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle> ingress_handle)>;

    template <typename DataTypeT>
    static void register_data_adapters() {
        if (!srf::node::EdgeAdapterRegistry::has_source_adapter(typeid(DataTypeT)))
        {
            std::type_index source_type = typeid(DataTypeT);
            VLOG(2) << "Registering PySRF source adapter for: " << type_name<DataTypeT>() << " "
                    << source_type.hash_code();
            node::EdgeAdapterRegistry::register_source_adapter(typeid(DataTypeT),
                                                               EdgeAdapterUtil::build_source_adapter<DataTypeT>());
        }

        if (!srf::node::EdgeAdapterRegistry::has_sink_adapter(typeid(DataTypeT)))
        {
            std::type_index sink_type = typeid(DataTypeT);
            VLOG(2) << "Registering PySRF sink adapter for: " << type_name<DataTypeT>() << " " << sink_type.hash_code();
            node::EdgeAdapterRegistry::register_sink_adapter(typeid(DataTypeT),
                                                             EdgeAdapterUtil::build_sink_adapter<DataTypeT>());
        }
    }

    template <typename InputT>
    static sink_adapter_fn_t build_sink_adapter()
    {
        return [](std::type_index source_type,
                  srf::node::SinkPropertiesBase& sink,
                  std::shared_ptr<channel::IngressHandle> ingress_handle) {
            if (source_type == typeid(PyHolder))
            {
                // Check to see if we have a conversion in pybind11
                if (pybind11::detail::get_type_info(sink.sink_type(true), false))
                {
                    // Shortcut the check to the registered converters
                    auto edge = std::make_shared<node::Edge<PyHolder, InputT>>(
                        std::dynamic_pointer_cast<channel::Ingress<InputT>>(ingress_handle));

                    // Using auto here confuses the lambda's return type with what's returned from
                    // ingress_for_source_type
                    std::shared_ptr<channel::IngressHandle> handle =
                        std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(edge);

                    return handle;
                }
            }

            return std::shared_ptr<channel::IngressHandle>(nullptr);
        };
    }

    template <typename OutputT>
    static source_adapter_fn_t build_source_adapter()
    {
        return [](srf::node::SourcePropertiesBase& source,
                  srf::node::SinkPropertiesBase& sink,
                  std::shared_ptr<channel::IngressHandle> ingress_handle) {
            // First check if there was a defined converter
            if (node::EdgeRegistry::has_converter(source.source_type(), sink.sink_type()))
            {
                return std::shared_ptr<channel::IngressHandle>(nullptr);
            }

            // Check here to see if we can short circuit if both of the types are the same
            if (source.source_type(false) == sink.sink_type(false))
            {
                // Register an edge identity converter
                node::IdentityEdgeConnector<OutputT>::register_converter();

                return std::shared_ptr<channel::IngressHandle>(nullptr);
            }

            // By this point several things have happened:
            // 1. Simple shortcut for matching types has failed. SourceT != SinkT
            // 2. We do not have a registered converter
            // 3. Both of our nodes are registered python nodes, but their source and sink types may not be registered

            // We can come up with an edge if one of the following is true:
            // 1. The source is a pybind11::object and the sink is registered with pybind11
            // 2. The sink is a pybind11::object and the source is registered with pybind11
            // 3. Neither is a pybind11::object but both types are registered with pybind11 (worst case, C++ -> py ->
            // C++)

            auto writer_type = source.source_type(true);
            auto reader_type = sink.sink_type(true);

            // Check registrations with pybind11
            auto* writer_typei = pybind11::detail::get_type_info(source.source_type(true), false);
            auto* reader_typei = pybind11::detail::get_type_info(sink.sink_type(true), false);

            // Check if the source is a pybind11::object
            if (writer_type == typeid(PyHolder) && reader_typei)
            {
                return srf::node::EdgeBuilder::ingress_for_source_type(source.source_type(), sink, ingress_handle);
                // return sink_ingress_adapter_for_source_type(sink, writer_type);
            }

            // Check if the sink is a py::object
            if (reader_type == typeid(PyHolder) && writer_typei)
            {
                // TODO(MDD): To avoid a compound edge here, register an edge converter between OutputT and py::object
                node::EdgeConnector<OutputT, PyHolder>::register_converter();

                // Build the edge with the holder type
                return srf::node::EdgeBuilder::ingress_for_source_type(source.source_type(), sink, ingress_handle);
            }

            // Check if both have been registered with pybind 11
            if (writer_typei && reader_typei)
            {
                // TODO(MDD): Check python types to see if they are compatible

                // Py types can be converted but need a compound edge. Build that here
                // auto py_to_sink_edge = std::dynamic_pointer_cast<channel::Ingress<pybind11::object>>(
                //    sink.ingress_for_source_type(typeid(pybind11::object)));
                auto py_to_sink_edge = std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(
                    srf::node::EdgeBuilder::ingress_for_source_type(typeid(PyHolder), sink, ingress_handle));

                auto source_to_py_edge = std::make_shared<node::Edge<OutputT, PyHolder>>(py_to_sink_edge);

                LOG(WARNING) << "WARNING: A slow edge connection between C++ nodes '" << source.source_type_name()
                             << "' and '" << sink.sink_type_name()
                             << "' has been detected. Performance between "
                                "these nodes can be improved by registering an EdgeConverter at compile time. Without "
                                "this, conversion "
                                "to an intermediate python type will be necessary (i.e. C++ -> Python -> C++).";

                return std::dynamic_pointer_cast<channel::IngressHandle>(source_to_py_edge);
            }

            // Otherwise return base which most likely will fail
            return std::shared_ptr<channel::IngressHandle>(nullptr);
        };
    }
};

/**
 * @brief Sources which inherit this object will automatically attempt to register a pySRF adapter for their data type
 * with the EdgeAdaptorRegistry
 * @tparam SourceT Data type the inheriting source emits
 */
template <typename SourceT>
struct AutoRegSourceAdapter
{
    AutoRegSourceAdapter()
    {
        // force register_adapter to be called, once, the first time a derived object is constructed.
        static bool _init = register_adapter();
    }

    static bool register_adapter()
    {
        if (!srf::node::EdgeAdapterRegistry::has_source_adapter(typeid(SourceT)))
        {
            std::type_index source_type = typeid(SourceT);
            VLOG(2) << "Registering PySRF source adapter for: " << type_name<SourceT>() << " "
                    << source_type.hash_code();
            node::EdgeAdapterRegistry::register_source_adapter(typeid(SourceT),
                                                               EdgeAdapterUtil::build_source_adapter<SourceT>());
        }

        return true;
    }
};

/**
 * @brief Sinks which inherit this object will automatically attempt to register a pySRF adapter for their data type
 * with the EdgeAdaptorRegistry
 * @tparam SinkT Data type the inheriting sink receives
 */
template <typename SinkT>
struct AutoRegSinkAdapter
{
    AutoRegSinkAdapter()
    {
        // force register_adapter to be called, once, the first time a derived object is constructed.
        static bool _init = register_adapter();
    }

    static bool register_adapter()
    {
        if (!srf::node::EdgeAdapterRegistry::has_sink_adapter(typeid(SinkT)))
        {
            std::type_index sink_type = typeid(SinkT);
            VLOG(2) << "Registering PySRF sink adapter for: " << type_name<SinkT>() << " " << sink_type.hash_code();
            node::EdgeAdapterRegistry::register_sink_adapter(typeid(SinkT),
                                                             EdgeAdapterUtil::build_sink_adapter<SinkT>());
        }

        return true;
    }
};
}  // namespace srf::pysrf
