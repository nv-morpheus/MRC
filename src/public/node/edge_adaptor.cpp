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

#include <srf/node/edge_adaptor.hpp>

#include <srf/channel/ingress.hpp>
#include <srf/node/edge_registry.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>

namespace srf::node {
std::shared_ptr<srf::channel::IngressHandle> DefaultEdgeAdaptor::try_construct_ingress(
    SourcePropertiesBase& source, SinkPropertiesBase& sink, std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    auto fn_converter = EdgeRegistry::find_converter(source.source_type(), sink.sink_type());
    return fn_converter(ingress_handle);
}
}  // namespace srf::node