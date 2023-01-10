/**
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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/forward.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge.hpp"
#include "mrc/node/edge_adapter_registry.hpp"
#include "mrc/type_traits.hpp"

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
template <typename InputT, typename OutputT>
struct EdgeConnector
{
  public:
    EdgeConnector() = delete;

    static void register_converter()
    {
        EdgeAdapterRegistry::register_ingress_converter(
            typeid(InputT), typeid(OutputT), [](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<OutputT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<OutputT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                // Build a new connector
                return std::make_shared<ConvertingEdgeWritable<InputT, OutputT>>(std::move(ingress));
            });

        EdgeAdapterRegistry::register_egress_converter(
            typeid(InputT), typeid(OutputT), [](std::shared_ptr<IEdgeReadableBase> channel) {
                std::shared_ptr<IEdgeReadable<InputT>> egress =
                    std::dynamic_pointer_cast<IEdgeReadable<InputT>>(channel);

                DCHECK(egress) << "Channel is not an egress of the correct type";

                // Build a new connector
                return std::make_shared<ConvertingEdgeReadable<InputT, OutputT>>(std::move(egress));
            });
    }

    static void register_converter(std::function<OutputT(InputT&&)> lambda_fn)
    {
        EdgeAdapterRegistry::register_ingress_converter(
            typeid(InputT), typeid(OutputT), [lambda_fn](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<OutputT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<OutputT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                // Build a new connector
                return std::make_shared<LambdaConvertingEdgeWritable<InputT, OutputT>>(lambda_fn, std::move(ingress));
            });

        EdgeAdapterRegistry::register_egress_converter(
            typeid(InputT), typeid(OutputT), [lambda_fn](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeReadable<InputT>> egress =
                    std::dynamic_pointer_cast<IEdgeReadable<InputT>>(channel);

                DCHECK(egress) << "Channel is not an egress of the correct type";

                // Build a new connector
                return std::make_shared<LambdaConvertingEdgeReadable<InputT, OutputT>>(lambda_fn, std::move(egress));
            });
    }

    static void register_dynamic_cast_converter()
    {
        EdgeAdapterRegistry::register_ingress_converter(
            typeid(InputT), typeid(OutputT), [](std::shared_ptr<IEdgeWritableBase> channel) {
                std::shared_ptr<IEdgeWritable<OutputT>> ingress =
                    std::dynamic_pointer_cast<IEdgeWritable<OutputT>>(channel);

                DCHECK(ingress) << "Channel is not an ingress of the correct type";

                if constexpr (is_shared_ptr_v<InputT> && is_shared_ptr_v<OutputT>)
                {
                    using sink_unwrapped_t = typename OutputT::element_type;

                    return std::make_shared<LambdaConvertingEdgeWritable<InputT, OutputT>>(
                        [](InputT&& data) {
                            // Call dynamic conversion on the shared_ptr
                            return std::dynamic_pointer_cast<sink_unwrapped_t>(data);
                        },
                        std::move(ingress));
                }
                else
                {
                    return std::make_shared<LambdaConvertingEdgeWritable<InputT, OutputT>>(
                        [](InputT&& data) {
                            // Normal dynamic_cast
                            return dynamic_cast<OutputT>(data);
                        },
                        std::move(ingress));
                }
            });

        EdgeAdapterRegistry::register_egress_converter(
            typeid(InputT), typeid(OutputT), [](std::shared_ptr<IEdgeReadableBase> channel) {
                std::shared_ptr<IEdgeReadable<InputT>> egress =
                    std::dynamic_pointer_cast<IEdgeReadable<InputT>>(channel);

                DCHECK(egress) << "Channel is not an egress of the correct type";

                if constexpr (is_shared_ptr_v<InputT> && is_shared_ptr_v<OutputT>)
                {
                    using sink_unwrapped_t = typename OutputT::element_type;

                    return std::make_shared<LambdaConvertingEdgeReadable<InputT, OutputT>>(
                        [](InputT&& data) {
                            // Call dynamic conversion on the shared_ptr
                            return std::dynamic_pointer_cast<sink_unwrapped_t>(data);
                        },
                        std::move(egress));
                }
                else
                {
                    return std::make_shared<LambdaConvertingEdgeReadable<InputT, OutputT>>(
                        [](InputT&& data) {
                            // Normal dynamic_cast
                            return dynamic_cast<OutputT>(data);
                        },
                        std::move(egress));
                }
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
        EdgeAdapterRegistry::register_ingress_converter(
            typeid(T), typeid(T), [](std::shared_ptr<IEdgeWritableBase> channel) { return channel; });
        EdgeAdapterRegistry::register_egress_converter(
            typeid(T), typeid(T), [](std::shared_ptr<IEdgeReadableBase> channel) { return channel; });
    }
};

}  // namespace mrc::node
