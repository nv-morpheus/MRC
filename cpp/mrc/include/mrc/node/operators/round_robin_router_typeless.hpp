/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/edge/deferred_edge.hpp"

#include <atomic>
#include <memory>
#include <mutex>

namespace mrc::node {

class RoundRobinRouterTypeless : public edge::IWritableProviderBase, public edge::IWritableAcceptorBase
{
  public:
    std::shared_ptr<edge::WritableEdgeHandle> get_writable_edge_handle() const override
    {
        auto* self = const_cast<RoundRobinRouterTypeless*>(this);

        // Create a new upstream edge. On connection, have it attach to any downstreams
        auto deferred_ingress = std::make_shared<edge::DeferredWritableEdgeHandle>(
            [self](std::shared_ptr<edge::DeferredWritableMultiEdgeBase> deferred_edge) {
                // Set the broadcast indices function
                deferred_edge->set_indices_fn([self](edge::DeferredWritableMultiEdgeBase& deferred_edge) {
                    // Increment the index and return the key for that index
                    auto next_idx = self->m_current_idx++;

                    auto current_keys = deferred_edge.edge_connection_keys();

                    return std::vector<size_t>{current_keys[next_idx % current_keys.size()]};
                });

                // Need to work with weak ptr here otherwise we will keep it from closing
                std::weak_ptr<edge::DeferredWritableMultiEdgeBase> weak_deferred_edge = deferred_edge;

                // Use a connector here in case the object never gets set to an edge
                deferred_edge->add_connector([self, weak_deferred_edge]() {
                    // Lock whenever working on the handles
                    std::unique_lock<std::mutex> lock(self->m_mutex);

                    // Save to the upstream handles
                    self->m_upstream_handles.emplace_back(weak_deferred_edge);

                    auto deferred_edge = weak_deferred_edge.lock();

                    CHECK(deferred_edge) << "Edge was destroyed before making connection.";

                    for (const auto& downstream : self->m_downstream_handles)
                    {
                        auto count = deferred_edge->edge_connection_count();

                        // Connect
                        deferred_edge->set_writable_edge_handle(count, downstream);
                    }

                    // Now add a disconnector that will remove it from the list
                    deferred_edge->add_disconnector([self, weak_deferred_edge]() {
                        // Need to lock here since this could be driven by different progress engines
                        std::unique_lock<std::mutex> lock(self->m_mutex);

                        bool is_expired = weak_deferred_edge.expired();

                        // Cull all expired ptrs from the list
                        auto iter = self->m_upstream_handles.begin();

                        while (iter != self->m_upstream_handles.end())
                        {
                            if ((*iter).expired())
                            {
                                iter = self->m_upstream_handles.erase(iter);
                            }
                            else
                            {
                                ++iter;
                            }
                        }

                        // If there are no more upstream handles, then delete the downstream
                        if (self->m_upstream_handles.empty())
                        {
                            self->m_downstream_handles.clear();
                        }
                    });
                });
            });

        return deferred_ingress;
    }

    edge::EdgeTypeInfo writable_provider_type() const override
    {
        return edge::EdgeTypeInfo::create_deferred();
    }

    void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    {
        // Lock whenever working on the handles
        std::unique_lock<std::mutex> lock(m_mutex);

        // We have a new downstream object. Hold onto it
        m_downstream_handles.push_back(ingress);

        // If we have an upstream object, try to make a connection now
        for (auto& upstream_weak : m_upstream_handles)
        {
            auto upstream = upstream_weak.lock();

            CHECK(upstream) << "Upstream edge went out of scope before downstream edges were connected";

            auto count = upstream->edge_connection_count();

            // Connect
            upstream->set_writable_edge_handle(count, ingress);
        }
    }

    edge::EdgeTypeInfo writable_acceptor_type() const override
    {
        return edge::EdgeTypeInfo::create_deferred();
    }

  private:
    std::mutex m_mutex;
    std::atomic_size_t m_current_idx{0};
    std::vector<std::weak_ptr<edge::DeferredWritableMultiEdgeBase>> m_upstream_handles;
    std::vector<std::shared_ptr<edge::WritableEdgeHandle>> m_downstream_handles;
};

}  // namespace mrc::node
