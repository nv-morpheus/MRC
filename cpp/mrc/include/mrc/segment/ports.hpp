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

#pragma once

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/segment/egress_port.hpp"

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace mrc::segment {

class Definition;

template <typename BaseT>
struct PortInfo
{
    using port_builder_fn_t = std::function<std::shared_ptr<BaseT>(const SegmentAddress&, const PortName&)>;

    PortInfo(std::string n, std::type_index t, port_builder_fn_t builder) :
      name(std::move(n)),
      type_index(std::move(t)),
      port_builder_fn(std::move(builder))
    {}

    // Port name
    std::string name;

    // Type info
    std::type_index type_index;

    // Function that builds the port object
    port_builder_fn_t port_builder_fn;
};

template <typename BaseT>
class Ports
{
  public:
    using port_info_t       = PortInfo<BaseT>;
    using port_builder_fn_t = std::function<std::shared_ptr<BaseT>(const SegmentAddress&, const PortName&)>;

    Ports() = default;

    Ports(std::vector<std::shared_ptr<port_info_t>> port_infos)
    {
        for (const auto& info : port_infos)
        {
            CHECK(!m_info.contains(info->name)) << "Duplicate port name '" << info->name << "' detected";

            m_info[info->name] = info;
        }
        // if (names.size() != builder_fns.size())
        // {
        //     LOG(ERROR) << "expected " << builder_fns.size() << " port names; got " << names.size();
        //     throw exceptions::MrcRuntimeError("invalid number of port names");
        // }

        // // test for uniqueness
        // std::set<std::string> unique_names(names.begin(), names.end());
        // if (unique_names.size() != builder_fns.size())
        // {
        //     LOG(ERROR) << "error: port names must be unique";
        //     throw exceptions::MrcRuntimeError("port names must be unique");
        // }

        // // store names
        // m_names = names;

        // for (int i = 0; i < names.size(); ++i)  // NOLINT
        // {
        //     auto builder         = builder_fns[i];
        //     auto name            = names[i];
        //     m_initializers[name] = [builder, name](const SegmentAddress& address) {
        //         return builder(address, name);
        //     };
        // }
    }

    // Ports(std::vector<std::string> names, std::vector<port_builder_fn_t> builder_fns)
    // {
    //     if (names.size() != builder_fns.size())
    //     {
    //         LOG(ERROR) << "expected " << builder_fns.size() << " port names; got " << names.size();
    //         throw exceptions::MrcRuntimeError("invalid number of port names");
    //     }

    //     // test for uniqueness
    //     std::set<std::string> unique_names(names.begin(), names.end());
    //     if (unique_names.size() != builder_fns.size())
    //     {
    //         LOG(ERROR) << "error: port names must be unique";
    //         throw exceptions::MrcRuntimeError("port names must be unique");
    //     }

    //     // store names
    //     m_names = names;

    //     for (int i = 0; i < names.size(); ++i)  // NOLINT
    //     {
    //         auto builder         = builder_fns[i];
    //         auto name            = names[i];
    //         m_initializers[name] = [builder, name](const SegmentAddress& address) {
    //             return builder(address, name);
    //         };
    //     }
    // }

    std::vector<std::string> names() const
    {
        std::vector<std::string> names;

        for (const auto& [name, info] : m_info)
        {
            names.push_back(name);
        }

        return names;
    }

    std::vector<std::type_index> type_indices() const
    {
        std::vector<std::type_index> type_indices;

        for (const auto& [name, info] : m_info)
        {
            type_indices.push_back(info.type_index);
        }

        return type_indices;
    }

    // const std::vector<std::string>& names() const
    // {
    //     return m_names;
    // }

  private:
    // const std::map<std::string, std::function<std::shared_ptr<BaseT>(const SegmentAddress&)>>& get_initializers()
    // const
    // {
    //     return m_initializers;
    // }

    const std::map<std::string, std::shared_ptr<const port_info_t>>& get_info() const
    {
        return m_info;
    }

    std::map<std::string, std::shared_ptr<const port_info_t>> m_info;

    friend Definition;
    friend class SegmentDefinition;
};

}  // namespace mrc::segment
