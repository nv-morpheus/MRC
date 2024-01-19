/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/edge/edge_builder.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <memory>
#include <random>

namespace mrc::manifold {

struct EgressDelegate
{
    virtual ~EgressDelegate()                                                                        = default;
    virtual void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) = 0;
    virtual void shutdown(){};
};

template <typename T>
class TypedEgress : public EgressDelegate
{
  public:
    void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) final
    {
        auto sink = dynamic_cast<edge::IWritableProvider<T>*>(output_sink);
        CHECK(sink);
        do_add_output(address, sink);
    }

  private:
    virtual void do_add_output(const SegmentAddress& address, edge::IWritableProvider<T>* output_sink) = 0;
};

template <typename T>
class RoundRobinEgress : public node::Router<SegmentAddress, T>, public TypedEgress<T>
{
  public:
    void shutdown() final
    {
        DVLOG(10) << "Releasing edges from manifold egress";
        node::Router<SegmentAddress, T>::release_edge_connections();
    }

  protected:
    SegmentAddress determine_key_for_value(const T& t) override
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
    void do_add_output(const SegmentAddress& address, edge::IWritableProvider<T>* sink) override
    {
        mrc::make_edge(*this->get_source(address), *sink);
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

}  // namespace mrc::manifold
