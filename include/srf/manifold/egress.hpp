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

#include "srf/manifold/interface.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/muxer.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/types.hpp"

#include <cstddef>
#include <memory>
#include <random>

namespace srf::manifold {

struct EgressDelegate
{
    virtual ~EgressDelegate() = default;
    virtual void add_output(const SegmentAddress& address, std::shared_ptr<node::IIngressProviderBase> output_sink) = 0;
};

template <typename T>
class TypedEgress : public EgressDelegate
{
  public:
    void add_output(const SegmentAddress& address, std::shared_ptr<node::IIngressProviderBase> output_sink) final
    {
        auto sink = std::dynamic_pointer_cast<node::IIngressProvider<T>>(output_sink);
        CHECK(sink);
        do_add_output(address, sink);
    }

  private:
    virtual void do_add_output(const SegmentAddress& address,
                               std::shared_ptr<node::IIngressProvider<T>> output_sink) = 0;
};

// template <typename T>
// class MappedEgress : public TypedEgress<T>
// {
//   public:
//     using channel_map_t = std::unordered_map<SegmentAddress, std::shared_ptr<node::EdgeWritable<T>>>;

//     // const channel_map_t& output_channels() const
//     // {
//     //     return m_outputs;
//     // }

//     // void clear()
//     // {
//     //     m_outputs.clear();
//     // }

//   protected:
//     void do_add_output(const SegmentAddress& address, std::shared_ptr<node::IIngressProvider<T>> sink) override
//     {
//         // auto search = m_outputs.find(address);
//         // CHECK(search == m_outputs.end());
//         // auto output_channel = std::make_unique<node::SourceChannelWriteable<T>>();
//         // node::make_edge(*output_channel, sink);
//         // m_outputs[address] = std::move(output_channel);
//         this->set_edge(address, sink->get_ingress());
//     }

//   private:
//     // std::unordered_map<SegmentAddress, std::unique_ptr<node::SourceChannelWriteable<T>>> m_outputs;
// };

template <typename T>
class RoundRobinEgress : public node::Router<SegmentAddress, T>, public TypedEgress<T>
{
  protected:
    virtual SegmentAddress determine_key_for_value(const T& t)
    {
        CHECK_LT(m_next, m_pick_list.size());
        auto next = m_next++;
        // roll counter before await_write which could yield
        if (m_next == m_pick_list.size())
        {
            m_next = 0;
        }

        return m_pick_list[next];
    }

  private:
    void do_add_output(const SegmentAddress& address, std::shared_ptr<node::IIngressProvider<T>> sink) override
    {
        node::make_edge(*this->get_source(address), *sink);
        update_pick_list();
    }

    void update_pick_list()
    {
        // Make a copy of the keys
        m_pick_list = this->edge_connection_keys();

        // Shuffle the keys
        std::shuffle(m_pick_list.begin(), m_pick_list.end(), std::mt19937(std::random_device()()));
        m_next = 0;
    }

    std::size_t m_next{0};
    std::vector<SegmentAddress> m_pick_list;
};

}  // namespace srf::manifold
