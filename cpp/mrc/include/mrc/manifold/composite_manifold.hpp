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

#include "mrc/manifold/egress.hpp"
#include "mrc/manifold/ingress.hpp"
#include "mrc/manifold/manifold.hpp"
#include "mrc/segment/utils.hpp"

namespace mrc::manifold {

template <typename IngressT, typename EgressT>
class CompositeManifold : public Manifold
{
    static_assert(std::is_base_of_v<IngressDelegate, IngressT>, "ingress must be derived from IngressDelegate");
    static_assert(std::is_base_of_v<EgressDelegate, EgressT>, "ingress must be derived from EgressDelegate");

  public:
    CompositeManifold(PortName port_name, pipeline::Resources& resources) : Manifold(std::move(port_name), resources)
    {
        // construct IngressT and EgressT on the NUMA node / memory domain in which the object will run
        this->resources()
            .main()
            .enqueue([this] {
                m_ingress = std::make_unique<IngressT>();
                m_egress  = std::make_unique<EgressT>();
            })
            .get();
    }
    CompositeManifold(PortName port_name,
                      pipeline::Resources& resources,
                      std::unique_ptr<IngressT> ingress,
                      std::unique_ptr<EgressT> egress) :
      Manifold(std::move(port_name), resources),
      m_ingress(std::move(ingress)),
      m_egress(std::move(egress))
    {}

  protected:
    IngressT& ingress()
    {
        CHECK(m_ingress);
        return *m_ingress;
    }

    EgressT& egress()
    {
        CHECK(m_egress);
        return *m_egress;
    }

  private:
    void do_add_input(const SegmentAddress& address, node::SourcePropertiesBase* input_source) final
    {
        // enqueue update to be done later
        m_input_updates.push_back([this, address, input_source] {
            DVLOG(10) << info() << ": ingress attaching to upstream segment " << segment::info(address);
            m_ingress->add_input(address, input_source);
            on_add_input(address);
        });
    }

    void do_add_output(const SegmentAddress& address, node::SinkPropertiesBase* output_sink) final
    {
        // enqueue update to be done later
        m_output_updates.push_back([this, address, output_sink] {
            DVLOG(10) << info() << ": egress attaching to downstream segment " << segment::info(address);
            m_egress->add_output(address, output_sink);
            on_add_output(address);
        });
    }

    void update(std::vector<std::function<void()>>& updates)
    {
        resources()
            .main()
            .enqueue([&] {
                for (auto& update_fn : updates)
                {
                    update_fn();
                }
            })
            .get();
        updates.clear();
    }

    void update_inputs() final
    {
        will_update_inputs();
        if (!m_input_updates.empty())
        {
            DVLOG(10) << info() << ": issuing all enqueued input updates";
            update(m_input_updates);
            DVLOG(10) << port_name() << " manifold finished input updates";
        }
    }

    void update_outputs() final
    {
        will_update_outputs();
        if (!m_output_updates.empty())
        {
            DVLOG(10) << info() << ": issuing all enqueued output updates";
            update(m_output_updates);
            DVLOG(10) << port_name() << " manifold finished output updates";
        }
    }

    virtual void on_add_input(const SegmentAddress& address) {}
    virtual void on_add_output(const SegmentAddress& address) {}

    virtual void will_update_inputs() {}
    virtual void will_update_outputs() {}

    std::vector<std::function<void()>> m_input_updates;
    std::vector<std::function<void()>> m_output_updates;

    std::unique_ptr<IngressT> m_ingress;
    std::unique_ptr<EgressT> m_egress;
};

}  // namespace mrc::manifold
