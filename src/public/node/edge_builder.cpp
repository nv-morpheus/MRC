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

#include <srf/node/edge_builder.hpp>

#include <srf/node/edge_adaptor.hpp>
#include <srf/node/edge_registry.hpp>

#include <mutex>

namespace srf::node {
std::mutex EdgeBuilder::s_edge_adaptor_mutex{};
std::map<std::string, std::shared_ptr<EdgeAdaptorBase>> EdgeBuilder::s_edge_adaptors{
    {"default", std::make_shared<DefaultEdgeAdaptor>()}};

void EdgeBuilder::make_edge_typeless(SourcePropertiesBase& source,
                                     SinkPropertiesBase& sink,
                                     const std::string& constructor_id,
                                     bool allow_narrowing)
{
    auto iter_adaptors = s_edge_adaptors.find(constructor_id);
    if (iter_adaptors == s_edge_adaptors.end())
    {
        std::stringstream sstream;
        sstream << "Specified unkown edge adaptor: " + constructor_id;

        LOG(ERROR) << sstream.str();
        throw std::runtime_error(sstream.str());
    }

    source.complete_edge(iter_adaptors->second->try_construct_ingress(source, sink, sink.ingress_handle()));
}
}  // namespace srf::node