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

#include "mrc/manifold/interface.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

#include <memory>

namespace mrc::manifold {

struct IngressDelegate
{
    virtual ~IngressDelegate()                                                                      = default;
    virtual void add_input(const SegmentAddress& address, node::IIngressAcceptorBase* input_source) = 0;
};

template <typename T>
class TypedIngress : public IngressDelegate
{
  public:
    // node::IIngressAcceptor<T>& source()
    // {
    //     auto sink = std::dynamic_pointer_cast<node::IIngressAcceptor<T>>(this->source_base());
    //     CHECK(sink);
    //     return *sink;
    // }

    void add_input(const SegmentAddress& address, node::IIngressAcceptorBase* input_source) final
    {
        auto source = dynamic_cast<node::IIngressAcceptor<T>*>(input_source);
        CHECK(source);
        do_add_input(address, source);
    }

  private:
    // virtual node::IIngressAcceptorBase& source_base()                                                           = 0;
    virtual void do_add_input(const SegmentAddress& address, node::IIngressAcceptor<T>* source) = 0;
};

template <typename T>
class MuxedIngress : public node::Muxer<T>, public TypedIngress<T>
{
  public:
    // MuxedIngress() : m_muxer(std::make_shared<node::Muxer<T>>()) {}

  protected:
    void do_add_input(const SegmentAddress& address, node::IIngressAcceptor<T>* source) final
    {
        // source->set_ingress(this->get)
        node::make_edge(*source, *this);
    }

  private:
    // node::IIngressAcceptorBase& source_base() final
    // {
    //     return *m_muxer;
    // }

    // std::shared_ptr<node::Muxer<T>> m_muxer;
};

}  // namespace mrc::manifold
