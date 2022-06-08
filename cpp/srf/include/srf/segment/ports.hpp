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

#include <srf/exceptions/runtime_error.hpp>
#include <srf/segment/egress_port.hpp>

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace srf::segment {

class Definition;

template <template <class> class PortT, typename BaseT, typename... TypesT>
class Ports
{
  public:
    static constexpr auto PortCount = sizeof...(TypesT);
    using port_builder_fn_t         = std::function<std::shared_ptr<BaseT>(const SegmentAddress&, const PortName&)>;

    Ports(std::vector<std::string> names)
    {
        if (names.size() != PortCount)
        {
            LOG(ERROR) << "expected " << PortCount << " port names; got " << names.size();
            throw exceptions::SrfRuntimeError("invalid number of port names");
        }

        // test for uniqueness
        std::set<std::string> unique_names(names.begin(), names.end());
        if (unique_names.size() != PortCount)
        {
            LOG(ERROR) << "error: port names must be unique";
            throw exceptions::SrfRuntimeError("port names must be unique");
        }

        // store names
        m_names = names;

        std::vector<port_builder_fn_t> builders;
        (builders.push_back([](const SegmentAddress& address, const PortName& name) {
            return std::make_shared<PortT<TypesT>>(address, name);
        }),
         ...);

        if (builders.size() != PortCount)
        {
            throw exceptions::SrfRuntimeError("invalid number of initializers");
        }

        for (int i = 0; i < names.size(); ++i)  // NOLINT
        {
            auto builder         = builders[i];
            auto name            = names[i];
            m_initializers[name] = [builder, name](const SegmentAddress& address) { return builder(address, name); };
        }
    }

    constexpr auto port_count() const
    {
        return PortCount;
    }

    const std::vector<std::string>& names() const
    {
        return m_names;
    }

  private:
    std::vector<std::string> m_names;
    std::map<std::string, std::function<std::shared_ptr<BaseT>(const SegmentAddress&)>> m_initializers;
    friend Definition;
};

}  // namespace srf::segment
