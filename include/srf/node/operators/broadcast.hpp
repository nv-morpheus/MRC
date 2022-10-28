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

#include "srf/channel/status.hpp"
#include "srf/node/operators/operator.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/type_traits.hpp"

namespace srf::node {

// template <typename T>
// class Broadcast final : public Operator<T>
// {
//   public:
//     Broadcast(bool deep_copy = false) : m_deep_copy(deep_copy) {}
//     ~Broadcast() final = default;

//     /**
//      * @brief Provides a reference to a SourceChannel<T>; this should be captured or used immediately with
//      * node::make_edge
//      *
//      * @return SourceChannel<T>&
//      */
//     [[nodiscard]] SourceChannel<T>& make_source()
//     {
//         return m_downstream_channels.emplace_back();
//     }

//   private:
//     // Operator::on_next
//     inline channel::Status on_next(T&& data) final
//     {
//         for (int i = 1; i < m_downstream_channels.size(); ++i)
//         {
//             if constexpr (is_shared_ptr<T>::value)
//             {
//                 if (m_deep_copy)
//                 {
//                     auto deep_copy = std::make_shared<typename T::element_type>(*data);
//                     CHECK(m_downstream_channels[i].await_write(std::move(deep_copy)) == channel::Status::success);
//                     continue;
//                 }
//             }

//             T shallow_copy(data);
//             CHECK(m_downstream_channels[i].await_write(std::move(shallow_copy)) == channel::Status::success);
//         }

//         return m_downstream_channels[0].await_write(std::move(data));
//     }

//     // Operator::on_complete
//     void on_complete() final
//     {
//         m_downstream_channels.clear();
//     }

//     std::vector<SourceChannelWriteable<T>> m_downstream_channels;
//     bool m_deep_copy;
// };

template <typename T>
class Broadcast : public IngressProvider<T>, public IIngressAcceptor<T>
{
    class BroadcastEdge : public IEdgeWritable<T>, public MultiSourceProperties<T, size_t>
    {
      public:
        BroadcastEdge(Broadcast& parent, bool deep_copy) : m_parent(parent), m_deep_copy(deep_copy) {}

        ~BroadcastEdge()
        {
            m_parent.on_complete();
        }

        channel::Status await_write(T&& data) override
        {
            for (size_t i = this->edge_connection_count() - 1; i > 0; --i)
            {
                if constexpr (is_shared_ptr<T>::value)
                {
                    if (m_deep_copy)
                    {
                        auto deep_copy = std::make_shared<typename T::element_type>(*data);
                        CHECK(this->get_writable_edge(i)->await_write(std::move(deep_copy)) ==
                              channel::Status::success);
                        continue;
                    }
                }

                T shallow_copy(data);
                CHECK(this->get_writable_edge(i)->await_write(std::move(shallow_copy)) == channel::Status::success);
            }

            return this->get_writable_edge(0)->await_write(std::move(data));
        }

        void add_downstream(std::shared_ptr<IngressHandleObj> downstream)
        {
            auto edge_count = this->edge_connection_count();

            this->make_edge_connection(edge_count, downstream);
        }

      private:
        Broadcast& m_parent;
        bool m_deep_copy;
    };

  public:
    Broadcast(bool deep_copy = false)
    {
        auto edge = std::make_shared<BroadcastEdge>(*this, deep_copy);

        // Save to avoid casting
        m_edge = edge;

        IngressProvider<int>::init_edge(edge);
    }

    ~Broadcast()
    {
        // Debug print
        VLOG(10) << "Destroying TestBroadcast";
    }

    void set_ingress_obj(std::shared_ptr<IngressHandleObj> ingress) override
    {
        if (auto e = m_edge.lock())
        {
            e->add_downstream(ingress);
        }
        else
        {
            LOG(ERROR) << "Edge was destroyed";
        }
    }

    void on_complete()
    {
        VLOG(10) << "TestBroadcast completed";
    }

  private:
    std::weak_ptr<BroadcastEdge> m_edge;
};

}  // namespace srf::node
