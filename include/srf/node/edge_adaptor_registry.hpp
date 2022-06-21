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

#include <srf/channel/ingress.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>

#include <functional>
#include <map>
#include <memory>
#include <typeindex>

namespace srf::node {

/**
 * @brief EdgeRegistry is an IngressHandle which contains the necessary conversion method to facilitate the creation an
 * Ingress from the type_index of the reader and writer.
 */
struct EdgeAdaptorRegistry
{
    // Function to create the adaptor function
    using adaptor_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(
        srf::node::SourcePropertiesBase&, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle>)>;


    using sink_adaptor_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(
        std::type_index, srf::node::SinkPropertiesBase&, std::shared_ptr<channel::IngressHandle>)>;

    EdgeAdaptorRegistry() = delete;

    // To register a converter, supply the reader/writer types and a function for creating the converter
    static void register_source_adaptor(std::type_index source_type, adaptor_fn_t adaptor_fn);
    static void register_sink_adaptor(std::type_index sink_type, sink_adaptor_fn_t adaptor_fn);

    static bool has_source_adaptor(std::type_index source_type);
    static bool has_sink_adaptor(std::type_index sink_type);

    static adaptor_fn_t find_source_adaptor(std::type_index source_type);
    static sink_adaptor_fn_t find_sink_adaptor(std::type_index sink_type);

    static std::map<std::type_index, adaptor_fn_t> registered_source_adaptors;
    static std::map<std::type_index, sink_adaptor_fn_t> registered_sink_adaptors;
};

}  // namespace srf::node
