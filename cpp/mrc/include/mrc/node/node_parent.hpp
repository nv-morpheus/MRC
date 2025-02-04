/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace mrc::node {

template <typename ChildT>
class HomogeneousNodeParent
{
  public:
    using child_node_t = ChildT;

    virtual std::map<std::string, std::reference_wrapper<child_node_t>> get_children_refs(
        std::optional<std::string> child_name = std::nullopt) const = 0;
};

template <typename... TypesT>
class HeterogeneousNodeParent
{
  public:
    using child_types_t = std::tuple<TypesT...>;

    virtual std::tuple<std::pair<std::string, std::reference_wrapper<TypesT>>...> get_children_refs() const = 0;
};

}  // namespace mrc::node
