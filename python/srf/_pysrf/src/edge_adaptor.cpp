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

#include <pysrf/edge_adaptor.hpp>

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <ostream>

namespace srf::pysrf {

std::shared_ptr<channel::IngressHandle> PysrfEdgeAdaptor::try_construct_ingress_fallback(
    std::type_index source_type,
    srf::node::SinkPropertiesBase& sink_base,
    std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    auto fn_converter = srf::node::EdgeRegistry::find_converter(source_type, sink_base.sink_type());
    return fn_converter(ingress_handle);
}

std::shared_ptr<channel::IngressHandle> PysrfEdgeAdaptor::try_construct_ingress(srf::node::SourcePropertiesBase& source,
                                                                                srf::node::SinkPropertiesBase& sink,
                                                                                std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    if (node::EdgeRegistry::has_converter(source.source_type(), sink.sink_type()))
    {
        return try_construct_ingress_fallback(source.source_type(), sink, ingress_handle);
    }

    // Check here to see if we can short circuit if both of the types are the same
    if (source.source_type(false) == sink.sink_type(false))
    {
        // Register an edge identity converter
        node::IdentityEdgeConnector<PyHolder>::register_converter();

        return try_construct_ingress_fallback(source.source_type(), sink, ingress_handle);
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
        return try_construct_ingress_fallback(source.source_type(), sink, ingress_handle);
    }

    // Check if the sink is a py::object
    if (reader_type == typeid(PyHolder) && writer_typei)
    {
        // TODO(MDD): To avoid a compound edge here, register an edge converter between OutputT and py::object
        node::EdgeConnector<PyHolder, PyHolder>::register_converter();

        return try_construct_ingress_fallback(source.source_type(), sink, ingress_handle);
    }

    // Check if both have been registered with pybind 11
    if (writer_typei && reader_typei)
    {
        // TODO(MDD): Check python types to see if they are compatible

        // Py types can be converted but need a compound edge. Build that here
        auto py_to_sink_edge = std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(
            try_construct_ingress_fallback(typeid(PyHolder), sink, ingress_handle));

        auto source_to_py_edge = std::make_shared<node::Edge<PyHolder, PyHolder>>(py_to_sink_edge);

        LOG(WARNING) << "WARNING: A slow edge connection between C++ nodes '" << source.source_type_name() << "' and '"
                     << sink.sink_type_name()
                     << "' has been detected. Performance between "
                        "these nodes can be improved by registering an EdgeConverter at compile time. Without "
                        "this, conversion "
                        "to an intermediate python type will be necessary (i.e. C++ -> Python -> C++).";

        return std::dynamic_pointer_cast<channel::IngressHandle>(source_to_py_edge);
    }

    // Otherwise return base which most likely will fail
    return try_construct_ingress_fallback(source.source_type(), sink, ingress_handle);
}

}  // namespace srf::pysrf
