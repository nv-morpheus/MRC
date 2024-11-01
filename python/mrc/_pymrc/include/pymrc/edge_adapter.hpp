/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/types.hpp"
#include "pymrc/utils.hpp"

#include "mrc/channel/forward.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_adapter_registry.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_connector.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
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

namespace mrc::pymrc {

/**
 * @brief Utility struct which supports building pyMRC source/sink adapter functions which can be registered with
 * the EdgeAdapterRegistry.
 */
struct EdgeAdapterUtil
{
    using ingress_adapter_fn_t = edge::EdgeAdapterRegistry::ingress_adapter_fn_t;
    using egress_adapter_fn_t  = edge::EdgeAdapterRegistry::egress_adapter_fn_t;

    template <typename DataTypeT>
    static void register_data_adapters()
    {
        edge::EdgeAdapterRegistry::register_ingress_adapter(EdgeAdapterUtil::build_sink_ingress_adapter<DataTypeT>());
        edge::EdgeAdapterRegistry::register_ingress_adapter(EdgeAdapterUtil::build_source_ingress_adapter<DataTypeT>());

        edge::EdgeAdapterRegistry::register_egress_adapter(EdgeAdapterUtil::build_sink_egress_adapter<DataTypeT>());
        edge::EdgeAdapterRegistry::register_egress_adapter(EdgeAdapterUtil::build_source_egress_adapter<DataTypeT>());
    }

    /**
     * @brief Generates an ingress adaptor for sink objects capable of converting from PyHolder -> InputT
     */
    template <typename InputT>
    static ingress_adapter_fn_t build_sink_ingress_adapter()
    {
        return [](const edge::EdgeTypeInfo& target_type, std::shared_ptr<edge::IEdgeWritableBase> ingress_handle) {
            // First try to convert the ingress to our type
            auto typed_ingress = std::dynamic_pointer_cast<edge::IEdgeWritable<InputT>>(ingress_handle);

            if (!typed_ingress)
            {
                // Cant do anything about this ingress
                return std::shared_ptr<edge::WritableEdgeHandle>(nullptr);
            }

            auto ingress_type = ingress_handle->get_type();

            // Check to see if we have a conversion in pybind11
            if (pybind11::detail::get_type_info(ingress_type.unwrapped_type(), false))
            {
                // Check if we are targeting a python object
                if (target_type.full_type() != typeid(PyHolder))
                {
                    // If that failed, check if our target type is a python object for a potential slow conversion
                    if (pybind11::detail::get_type_info(target_type.unwrapped_type(), false))
                    {
                        // Make the foundation of a slow connection. Show warning here
                        LOG(WARNING) << "WARNING: A slow edge connection between C++ nodes '"
                                     << type_name(target_type.full_type()) << "' and '"
                                     << type_name(ingress_type.full_type())
                                     << "' has been detected. Performance between "
                                        "these nodes can be improved by registering an EdgeConverter at compile time. "
                                        "Without "
                                        "this, conversion "
                                        "to an intermediate python type will be necessary (i.e. C++ -> Python -> C++).";
                    }
                    else
                    {
                        // Cant make a slow connection
                        return std::shared_ptr<edge::WritableEdgeHandle>(nullptr);
                    }
                }

                // Create a conversion from our type to PyHolder
                auto edge = std::make_shared<edge::ConvertingEdgeWritable<PyHolder, InputT>>(typed_ingress);

                return std::make_shared<edge::WritableEdgeHandle>(edge);
            }

            return std::shared_ptr<edge::WritableEdgeHandle>(nullptr);
        };
    }

    /**
     * @brief Generates an ingress adaptor for source objects capable of converting from OutputT -> PyHolder
     */
    template <typename OutputT>
    static ingress_adapter_fn_t build_source_ingress_adapter()
    {
        static_assert(IsComplete<pybind11::detail::make_caster<OutputT>>::value,
                      "OutputT must have a pybind11::detail::make_caster specialization");

        return [](const edge::EdgeTypeInfo& target_type, std::shared_ptr<edge::IEdgeWritableBase> ingress_handle) {
            // Check to make sure we are targeting this type
            if (target_type != edge::EdgeTypeInfo::create<OutputT>())
            {
                return std::shared_ptr<edge::WritableEdgeHandle>(nullptr);
            }

            auto ingress_type = ingress_handle->get_type();

            // Check if we are coming from a python object. Skip if we are not
            if (ingress_type.full_type() != typeid(PyHolder))
            {
                return std::shared_ptr<edge::WritableEdgeHandle>(nullptr);
            }

            // By the time we are here, there must be a C++ -> Python conversion that exists. So use that
            auto py_typed_ingress = std::dynamic_pointer_cast<edge::IEdgeWritable<PyHolder>>(ingress_handle);

            CHECK(py_typed_ingress) << "Invalid conversion. Incoming ingress is not a PyHolder";

            // Create a conversion from PyHolder to our type
            auto edge = std::make_shared<edge::ConvertingEdgeWritable<OutputT, PyHolder>>(py_typed_ingress);

            return std::make_shared<edge::WritableEdgeHandle>(edge);
        };
    }

    /**
     * @brief Generates an egress adaptor for sink objects capable of converting from PyHolder -> OutputT
     */
    template <typename OutputT>
    static egress_adapter_fn_t build_sink_egress_adapter()
    {
        return [](const edge::EdgeTypeInfo& target_type, std::shared_ptr<edge::IEdgeReadableBase> egress_handle) {
            // Check to make sure we are targeting this type
            if (target_type != edge::EdgeTypeInfo::create<OutputT>())
            {
                return std::shared_ptr<edge::ReadableEdgeHandle>(nullptr);
            }

            auto egress_type = egress_handle->get_type();

            // Check to see if we have a conversion in pybind11
            if (pybind11::detail::get_type_info(target_type.unwrapped_type(), false))
            {
                // Check if we are coming from a python object
                if (egress_type.full_type() == typeid(PyHolder))
                {
                    auto py_typed_egress = std::dynamic_pointer_cast<edge::IEdgeReadable<PyHolder>>(egress_handle);

                    CHECK(py_typed_egress) << "Invalid conversion. Incoming egress is not a PyHolder";

                    // Create a conversion from PyHolder to our type
                    auto edge = std::make_shared<edge::ConvertingEdgeReadable<PyHolder, OutputT>>(py_typed_egress);

                    return std::make_shared<edge::ReadableEdgeHandle>(edge);
                }
            }

            return std::shared_ptr<edge::ReadableEdgeHandle>(nullptr);
        };
    }

    /**
     * @brief Generates an egress adaptor for source objects capable of converting from InputT -> PyHolder
     */
    template <typename InputT>
    static egress_adapter_fn_t build_source_egress_adapter()
    {
        return [](const edge::EdgeTypeInfo& target_type, std::shared_ptr<edge::IEdgeReadableBase> egress_handle) {
            // First try to convert the egress to our type
            auto typed_egress = std::dynamic_pointer_cast<edge::IEdgeReadable<InputT>>(egress_handle);

            if (!typed_egress)
            {
                // Cant do anything about this egress
                return std::shared_ptr<edge::ReadableEdgeHandle>(nullptr);
            }

            auto egress_type = egress_handle->get_type();

            // Check to see if we have a conversion in pybind11
            if (pybind11::detail::get_type_info(egress_type.unwrapped_type(), false))
            {
                // Check if we are targeting a python object
                if (target_type.full_type() != typeid(PyHolder))
                {
                    // If that failed, check if our target type is a python object for a potential slow conversion
                    if (pybind11::detail::get_type_info(target_type.unwrapped_type(), false))
                    {
                        // Make the foundation of a slow connection. Show warning here
                        LOG(WARNING) << "WARNING: A slow edge connection between C++ nodes '"
                                     << type_name(target_type.full_type()) << "' and '"
                                     << type_name(egress_type.full_type())
                                     << "' has been detected. Performance between "
                                        "these nodes can be improved by registering an EdgeConverter at compile time. "
                                        "Without "
                                        "this, conversion "
                                        "to an intermediate python type will be necessary (i.e. C++ -> Python -> C++).";
                    }
                    else
                    {
                        // Cant make a slow connection
                        return std::shared_ptr<edge::ReadableEdgeHandle>(nullptr);
                    }
                }

                // Create a conversion from our type to PyHolder
                auto edge = std::make_shared<edge::ConvertingEdgeReadable<InputT, PyHolder>>(typed_egress);

                return std::make_shared<edge::ReadableEdgeHandle>(edge);
            }

            return std::shared_ptr<edge::ReadableEdgeHandle>(nullptr);
        };
    }
};

/**
 * @brief Sources which inherit this object will automatically attempt to register a pyMRC adapter for their data type
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
        edge::EdgeAdapterRegistry::register_ingress_adapter(EdgeAdapterUtil::build_source_ingress_adapter<SourceT>());
        edge::EdgeAdapterRegistry::register_egress_adapter(EdgeAdapterUtil::build_source_egress_adapter<SourceT>());

        return true;
    }
};

/**
 * @brief Sinks which inherit this object will automatically attempt to register a pyMRC adapter for their data type
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
        edge::EdgeAdapterRegistry::register_ingress_adapter(EdgeAdapterUtil::build_sink_ingress_adapter<SinkT>());
        edge::EdgeAdapterRegistry::register_egress_adapter(EdgeAdapterUtil::build_sink_egress_adapter<SinkT>());

        return true;
    }
};
}  // namespace mrc::pymrc
