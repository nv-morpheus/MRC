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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/forward.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/node/edge.hpp"
#include "mrc/node/edge_registry.hpp"

#include <glog/logging.h>

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>
#include <utility>

namespace mrc::node {

/**
 * @brief Convenience object to register Edge converters/adaptors with the EdgeRegistry
 *
 * @tparam SourceT
 * @tparam SinkT
 */
template <typename SourceT, typename SinkT>
struct EdgeConnector
{
  public:
    EdgeConnector() = delete;

    static void register_converter()
    {
        EdgeRegistry::register_converter(
            typeid(SourceT), typeid(SinkT), [](std::shared_ptr<channel::IngressHandle> channel) {
                std::shared_ptr<channel::Ingress<SinkT>> ingress =
                    std::dynamic_pointer_cast<channel::Ingress<SinkT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                // Build a new connector
                return std::make_shared<Edge<SourceT, SinkT>>(std::move(ingress));
            });
    }
};

/**
 * @brief Convenience class for performing a no-op edge conversion.
 */
template <typename T>
struct IdentityEdgeConnector
{
  public:
    IdentityEdgeConnector() = delete;

    static void register_converter()
    {
        EdgeRegistry::register_converter(
            typeid(T), typeid(T), [](std::shared_ptr<channel::IngressHandle> channel) { return channel; });
    }
};

}  // namespace mrc::node
