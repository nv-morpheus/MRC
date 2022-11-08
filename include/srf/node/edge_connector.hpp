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

#include "srf/channel/channel.hpp"
#include "srf/channel/forward.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/core/watcher.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_registry.hpp"
#include "srf/type_traits.hpp"

#include <glog/logging.h>

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>
#include <utility>

namespace srf::node {

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
            typeid(SourceT), typeid(SinkT), [](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<SinkT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<SinkT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                // Build a new connector
                return std::make_shared<ConvertingEdgeWritable<SourceT, SinkT>>(std::move(ingress));
            });
    }

    static void register_converter(typename LambdaConvertingEdgeWritable<SourceT, SinkT>::lambda_fn_t lambda_fn)
    {
        EdgeRegistry::register_converter(
            typeid(SourceT), typeid(SinkT), [lambda_fn](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<SinkT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<SinkT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                // Build a new connector
                return std::make_shared<LambdaConvertingEdgeWritable<SourceT, SinkT>>(lambda_fn, std::move(ingress));
            });
    }

    static void register_dynamic_cast_converter()
    {
        EdgeRegistry::register_converter(
            typeid(SourceT), typeid(SinkT), [](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<SinkT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<SinkT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                if constexpr (is_shared_ptr_v<SourceT> && is_shared_ptr_v<SinkT>)
                {
                    using sink_unwrapped_t = typename SinkT::element_type;

                    return std::make_shared<LambdaConvertingEdgeWritable<SourceT, SinkT>>(
                        [](SourceT&& data) {
                            // Call dynamic conversion on the shared_ptr
                            return std::dynamic_pointer_cast<sink_unwrapped_t>(data);
                        },
                        std::move(ingress));
                }
                else
                {
                    return std::make_shared<LambdaConvertingEdgeWritable<SourceT, SinkT>>(
                        [](SourceT&& data) {
                            // Normal dynamic_cast
                            return dynamic_cast<SinkT>(data);
                        },
                        std::move(ingress));
                }

                // Build a new connector
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
            typeid(T), typeid(T), [](std::shared_ptr<IEdgeWritableBase> channel) { return channel; });
    }
};

}  // namespace srf::node
