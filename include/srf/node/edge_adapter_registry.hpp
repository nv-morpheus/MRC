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

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <typeindex>

namespace srf::channel {
struct IngressHandle;
}

namespace srf::node {
class SinkPropertiesBase;
class SourcePropertiesBase;
}  // namespace srf::node

namespace srf::node {

/**
 * @brief EdgeAdaptorRegistry used for the registry of adapter routines which allow for customized runtime
 * edge construction and type deduction.
 *
 * Generally speaking, where an EdgeConverter defines a specific instance used to connect a compatible
 * source and sink, an EdgeAdaptor defines a process for identifying or creating an appropriate
 * EdgeConverter or set of EdgeConverters to make a source sink connection possible.
 */
struct EdgeAdapterRegistry
{
    // Function to create the adapter function
    using source_adapter_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(
        srf::node::SourcePropertiesBase&, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle>)>;

    using sink_adapter_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(
        std::type_index, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle>)>;

    EdgeAdapterRegistry() = delete;

    /**
     * @brief Registers an adapter function to be used for a given `(source|sink)_type`
     * @param source_type type index of the data type which will use `adapter_fn`
     * @param adapter_fn adapter function used to attempt to adapt a given source and sink
     */
    static void register_source_adapter(std::type_index source_type, source_adapter_fn_t adapter_fn);
    static void register_sink_adapter(std::type_index sink_type, sink_adapter_fn_t adapter_fn);

    /**
     * @brief Checks to see if an adapter is registered for a given type index
     * @param source_type
     * @return
     */
    static bool has_source_adapter(std::type_index source_type);
    static bool has_sink_adapter(std::type_index sink_type);

    /**
     * @brief Attempts to retrieve a source/sink adapter for a given type index
     * @param source_type
     * @return:
     */
    static source_adapter_fn_t find_source_adapter(std::type_index source_type);
    static sink_adapter_fn_t find_sink_adapter(std::type_index sink_type);

    static std::map<std::type_index, source_adapter_fn_t> registered_source_adapters;
    static std::map<std::type_index, sink_adapter_fn_t> registered_sink_adapters;

    static std::recursive_mutex s_mutex;
};
}  // namespace srf::node
