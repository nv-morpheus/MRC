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

#include "mrc/channel/ingress.hpp"

#include <functional>
#include <map>
#include <memory>
#include <typeindex>

namespace mrc::node {

/**
 * @brief EdgeRegistry is an IngressHandle which contains the necessary conversion method to facilitate the creation an
 * Ingress from the type_index of the reader and writer.
 */
struct EdgeRegistry
{
    // Function to create the edge converter
    using build_fn_t = std::function<std::shared_ptr<channel::IngressHandle>(std::shared_ptr<channel::IngressHandle>)>;

    EdgeRegistry() = delete;

    // To register a converter, supply the reader/writer types and a function for creating the converter
    static void register_converter(std::type_index writer_type, std::type_index reader_type, build_fn_t converter);

    static bool has_converter(std::type_index writer_type, std::type_index reader_type);

    static build_fn_t find_converter(std::type_index writer_type, std::type_index reader_type);

    // Goes from source type to sink type
    static std::map<std::type_index, std::map<std::type_index, build_fn_t>> registered_converters;
};

}  // namespace mrc::node
