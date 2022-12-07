/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/status.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/utils/type_utils.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace mrc::node {

class DeferredWritableMultiEdgeBase : public IMultiIngressAcceptorBase<std::size_t>,
                                      public virtual IEdgeWritableBase,
                                      public virtual EdgeTag
{
  public:
    using determine_indices_fn_t = std::function<std::vector<std::size_t>(DeferredWritableMultiEdgeBase&)>;

    virtual void set_indices_fn(determine_indices_fn_t indices_fn) = 0;

    virtual size_t edge_connection_count() const                  = 0;
    virtual std::vector<std::size_t> edge_connection_keys() const = 0;

  private:
};

struct DeferredIngressHandleObj : public IngressHandleObj
{
  public:
    using on_defer_fn_t = std::function<void(std::shared_ptr<DeferredWritableMultiEdgeBase>)>;

    DeferredIngressHandleObj(on_defer_fn_t on_connect) :
      IngressHandleObj(EdgeTypePair::create_deferred(), nullptr),
      m_on_connect(std::move(on_connect))
    {
        CHECK(m_on_connect) << "Must supply an on connect function";
    }

    bool is_deferred() const override
    {
        return true;
    }

  private:
    std::shared_ptr<IngressHandleObj> set_deferred_edge(std::shared_ptr<DeferredWritableMultiEdgeBase> deferred_edge)
    {
        // Call the on_connect function
        m_on_connect(deferred_edge);

        // this->set_ingress_handle(deferred_edge);

        return std::make_shared<IngressHandleObj>(deferred_edge);
    }

    // template <typename T>
    // std::shared_ptr<IngressHandleObj> make_deferred_edge()
    // {
    //     auto deferred_edge = std::make_shared<DeferredWritableMultiEdge<T>>();

    //     this->set_ingress_handle(deferred_edge);

    //     // Call the connection event
    //     if (m_on_connect)
    //     {
    //         m_on_connect(deferred_edge);
    //     }

    //     return std::make_shared<IngressHandleObj>(deferred_edge);
    // }

    DeferredWritableMultiEdgeBase::determine_indices_fn_t m_indices_fn{};
    on_defer_fn_t m_on_connect{};

    friend EdgeBuilder;
};

}  // namespace mrc::node
