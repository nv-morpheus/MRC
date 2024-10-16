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
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

#include <glog/logging.h>

#include <memory>

namespace mrc::manifold {

struct IngressDelegate
{
    virtual ~IngressDelegate()                                                                       = default;
    virtual void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) = 0;
    virtual void shutdown(){};
};

template <typename T>
class TypedIngress : public IngressDelegate
{
  public:
    void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) final
    {
        auto source = dynamic_cast<edge::IWritableAcceptor<T>*>(input_source);
        CHECK(source);
        do_add_input(address, source);
    }

  private:
    virtual void do_add_input(const SegmentAddress& address, edge::IWritableAcceptor<T>* source) = 0;
};

template <typename T>
class MuxedIngress : public node::Muxer<T>, public TypedIngress<T>
{
  public:
    void shutdown() final
    {
        DVLOG(10) << "Releasing edges from manifold ingress";
        node::SourceProperties<T>::release_edge_connection();
    }

  protected:
    void do_add_input(const SegmentAddress& address, edge::IWritableAcceptor<T>* source) final
    {
        // source->set_ingress(this->get)
        mrc::make_edge(*source, *this);
    }
};

}  // namespace mrc::manifold
