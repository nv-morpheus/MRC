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

#include "srf/types.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace srf::segment {
class EgressPortBase;
struct IngressPortBase;
struct ObjectProperties;
}  // namespace srf::segment

namespace srf::node {

/**
 * @brief Utility object used for constructing, inspecting, and manipulating ingress and egress ports dynamically.
 */
struct PortUtil
{
    using ingress_object_caster_fn_t =
        std::function<std::shared_ptr<segment::ObjectProperties>(std::shared_ptr<segment::IngressPortBase>)>;
    using egress_object_caster_fn_t =
        std::function<std::shared_ptr<segment::ObjectProperties>(std::shared_ptr<segment::EgressPortBase>)>;

    using ingress_builder_fn_t =
        std::function<std::shared_ptr<segment::IngressPortBase>(const SegmentAddress&, const PortName&)>;
    using egress_builder_fn_t =
        std::function<std::shared_ptr<segment::EgressPortBase>(const SegmentAddress&, const PortName&)>;

    // We store builder tuples to avoid ambiguities when attempting to construct objects via python.
    // While we have the ability to extract the underlying c++ type stored by a registered Pybind11 object,
    // we don't know apriori if we will be constructing a port of that data type or a shared_ptr to that
    // data type... so we register builders for both scenarios.
    using ingress_tuple_t = std::tuple<ingress_builder_fn_t, ingress_builder_fn_t>;
    using egress_tuple_t  = std::tuple<egress_builder_fn_t, egress_builder_fn_t>;

    using ingress_caster_tuple_t = std::tuple<ingress_object_caster_fn_t, ingress_object_caster_fn_t>;
    using egress_caster_tuple_t  = std::tuple<egress_object_caster_fn_t, egress_object_caster_fn_t>;

    PortUtil() = delete;

    PortUtil(std::type_index type_index);

    std::shared_ptr<segment::ObjectProperties> try_cast_ingress_base_to_object(
        std::shared_ptr<segment::IngressPortBase> base);

    std::shared_ptr<segment::ObjectProperties> try_cast_egress_base_to_object(
        std::shared_ptr<segment::EgressPortBase> base);

    const std::type_index m_port_data_type;

    // Builders for ingress/egress ports
    ingress_tuple_t m_ingress_builders{nullptr, nullptr};
    egress_tuple_t m_egress_builders{nullptr, nullptr};

    // Used to recover Source/SinkProperties from an ingress/egress base
    ingress_caster_tuple_t m_ingress_casters{nullptr, nullptr};
    egress_caster_tuple_t m_egress_casters{nullptr, nullptr};
};
/**
 * @brief Collection of static methods and data used to associate and retrieve PortUtil objects with std::type_index
 * values
 */
struct PortRegistry
{
    PortRegistry() = delete;

    /**
     * @brief Registers a PortUtil object, and updates our registered util map to associate it with its type_index
     * @param source_type type index of the data type which will use `adapter_fn`
     * @param adapter_fn adapter function used to attempt to adapt a given source and sink
     */
    static void register_port_util(std::shared_ptr<PortUtil> port_util);

    /**
     * @brief Checks to see if a PortUtil is registered for a given type index
     * @param type_index
     * @return true (exists) false otherwise
     */
    static bool has_port_util(std::type_index type_index);

    /**
     * @brief Attempts to retrieve a PortUtil object associated with the provided type_index
     * @param type_index : type_index which we expect to have an associated PortUtil
     * @return: Shared pointer to a PortUtil
     */
    static std::shared_ptr<PortUtil> find_port_util(std::type_index type_index);

    /**
     * @brief Associate string names with type indices; this can be used to make blind lookup casts when
     *  retrieving ports with dynamic constructors.
     * @param name
     * @param type_index
     */
    static void register_name_type_index_pair(std::string name, std::type_index type_index);
    static void register_name_type_index_pairs(std::vector<std::string> names,
                                               std::vector<std::type_index> type_indices);

    static std::map<std::type_index, std::shared_ptr<PortUtil>> s_registered_port_utils;

    static std::map<std::string, std::type_index> s_port_to_type_index;

    static std::recursive_mutex s_mutex;
};

}  // namespace srf::node
