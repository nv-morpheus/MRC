/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

namespace mrc::node {

template <typename InputT, typename OutputT = InputT>
class NodeComponent;

template <typename InputT, typename OutputT>
class NodeComponent : public ForwardingIngressProvider<InputT>, public IngressAcceptor<OutputT>
{
  public:
    NodeComponent() : ForwardingIngressProvider<InputT>() {}

    virtual ~NodeComponent() = default;

  protected:
    void on_complete() final
    {
        this->do_on_complete();

        SourceProperties<OutputT>::release_edge_connection();
    }

    virtual void do_on_complete()
    {
        // Nothing in base
    }
};

template <typename T>
class NodeComponent<T, T> : public ForwardingIngressProvider<T>, public IngressAcceptor<T>
{
  public:
    NodeComponent() : ForwardingIngressProvider<T>() {}

    virtual ~NodeComponent() = default;

  protected:
    channel::Status on_next(T&& t) override
    {
        return this->get_writable_edge()->await_write(std::move(t));
    }

    void on_complete() final
    {
        this->do_on_complete();

        SourceProperties<T>::release_edge_connection();
    }

    virtual void do_on_complete()
    {
        // Nothing in base
    }
};

template <typename InputT, typename OutputT = InputT>
class LambdaNodeComponent : public NodeComponent<InputT, OutputT>
{
  public:
    using on_next_fn_t     = std::function<OutputT(InputT&&)>;
    using on_complete_fn_t = std::function<void()>;

    LambdaNodeComponent(on_next_fn_t on_next_fn) : NodeComponent<InputT, OutputT>(), m_on_next_fn(std::move(on_next_fn))
    {}

    LambdaNodeComponent(on_next_fn_t on_next_fn, on_complete_fn_t on_complete_fn) :
      NodeComponent<InputT, OutputT>(),
      m_on_next_fn(std::move(on_next_fn)),
      m_on_complete_fn(std::move(on_complete_fn))
    {}

    virtual ~LambdaNodeComponent() = default;

  protected:
    channel::Status on_next(InputT&& t) override
    {
        return this->get_writable_edge()->await_write(m_on_next_fn(std::move(t)));
    }

    void do_on_complete() override
    {
        if (m_on_complete_fn)
        {
            m_on_complete_fn();
        }
    }

  private:
    on_next_fn_t m_on_next_fn;
    on_complete_fn_t m_on_complete_fn;
};

}  // namespace mrc::node
