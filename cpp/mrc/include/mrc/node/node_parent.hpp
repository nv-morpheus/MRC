/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mrc::node {

template <typename ChildT>
class DynamicNodeParent
{
  public:
    using child_node_t = ChildT;

    void add_child_sync_callback(std::function<void()> callback) {}

    virtual std::map<std::string, std::reference_wrapper<child_node_t>> get_children_refs(
        std::optional<std::string> child_name = std::nullopt) const = 0;

  protected:
    void children_updated() {}
};

template <typename... TypesT>
class NodeParent
{
  public:
    using child_types_t = std::tuple<TypesT...>;

    void add_child_sync_callback(std::function<void()> callback) {}

    virtual std::tuple<std::pair<std::string, std::reference_wrapper<TypesT>>...> get_children_refs() const = 0;
};

}  // namespace mrc::node
