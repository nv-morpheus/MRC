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

#include <srf/core/addresses.hpp>
#include <srf/segment/builder.hpp>
#include <srf/segment/definition.hpp>
#include <srf/segment/egress_ports.hpp>
#include <srf/segment/forward.hpp>
#include <srf/segment/ingress_ports.hpp>

namespace srf {

class Segment final
{
  public:
    template <typename... ArgsT>
    static std::shared_ptr<segment::Definition> create(std::string name, ArgsT&&... args)
    {
        return segment::Definition::create(std::move(name), std::forward<ArgsT>(args)...);
    }
};

}  // namespace srf
