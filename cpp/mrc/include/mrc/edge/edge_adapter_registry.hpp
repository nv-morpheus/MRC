/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

namespace mrc::edge {
class EdgeTypeInfo;
class IEdgeReadableBase;
class IEdgeWritableBase;
class ReadableEdgeHandle;
class WritableEdgeHandle;

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
    using ingress_converter_fn_t = std::function<std::shared_ptr<IEdgeWritableBase>(std::shared_ptr<IEdgeWritableBase>)>;

    using egress_converter_fn_t = std::function<std::shared_ptr<IEdgeReadableBase>(std::shared_ptr<IEdgeReadableBase>)>;

    using ingress_adapter_fn_t =
        std::function<std::shared_ptr<WritableEdgeHandle>(const EdgeTypeInfo&, std::shared_ptr<IEdgeWritableBase>)>;

    using egress_adapter_fn_t =
        std::function<std::shared_ptr<ReadableEdgeHandle>(const EdgeTypeInfo&, std::shared_ptr<IEdgeReadableBase>)>;

    EdgeAdapterRegistry() = delete;

    // To register a converter, supply the reader/writer types and a function for creating the converter
    static void register_ingress_converter(std::type_index input_type,
                                           std::type_index output_type,
                                           ingress_converter_fn_t converter_fn);

    static void register_egress_converter(std::type_index input_type,
                                          std::type_index output_type,
                                          egress_converter_fn_t converter_fn);

    static bool has_ingress_converter(std::type_index input_type, std::type_index output_type);

    static bool has_egress_converter(std::type_index input_type, std::type_index output_type);

    static ingress_converter_fn_t find_ingress_converter(std::type_index input_type, std::type_index output_type);

    static egress_converter_fn_t find_egress_converter(std::type_index input_type, std::type_index output_type);

    static void register_ingress_adapter(ingress_adapter_fn_t adapter_fn);

    static void register_egress_adapter(egress_adapter_fn_t adapter_fn);

    static const std::vector<ingress_adapter_fn_t>& get_ingress_adapters();

    static const std::vector<egress_adapter_fn_t>& get_egress_adapters();

  private:
    static std::map<std::type_index, std::map<std::type_index, ingress_converter_fn_t>> registered_ingress_converters;
    static std::map<std::type_index, std::map<std::type_index, egress_converter_fn_t>> registered_egress_converters;

    static std::vector<ingress_adapter_fn_t> registered_ingress_adapters;
    static std::vector<egress_adapter_fn_t> registered_egress_adapters;

    static std::recursive_mutex s_mutex;
};
}  // namespace mrc::edge
