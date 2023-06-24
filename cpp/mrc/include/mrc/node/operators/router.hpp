/*
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

#include "mrc/channel/status.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"

#include <map>
#include <memory>
#include <type_traits>

namespace mrc::node {

template <typename KeyT, typename OutputT>
class RouterWritableAcceptor : public MultiWritableAcceptor<KeyT, OutputT>
{
  public:
    using output_data_t = OutputT;

    RouterWritableAcceptor() = default;

    std::shared_ptr<edge::IWritableAcceptor<output_data_t>> get_source(const KeyT& key) const
    {
        // Simply return an object that will set the message to upstream and go away
        return std::make_shared<DownstreamEdge>(*const_cast<RouterWritableAcceptor<KeyT, OutputT>*>(this), key);
    }

    bool has_source(const KeyT& key) const
    {
        return MultiWritableAcceptor<KeyT, output_data_t>::get_edge_pair(key).first;
    }

    void drop_source(const KeyT& key)
    {
        MultiWritableAcceptor<KeyT, output_data_t>::release_edge_connection(key);
    }

    void drop_all_sources()
    {
        MultiWritableAcceptor<KeyT, output_data_t>::release_edge_connections();
    }

  protected:
    class DownstreamEdge : public edge::IWritableAcceptor<output_data_t>
    {
      public:
        DownstreamEdge(edge::IMultiWritableAcceptorBase<KeyT>& parent, KeyT key) :
          m_parent(parent),
          m_key(std::move(key))
        {}

        void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
        {
            m_parent.set_writable_edge_handle(m_key, std::move(ingress));
        }

      private:
        edge::IMultiWritableAcceptorBase<KeyT>& m_parent;
        KeyT m_key;
    };
};

template <typename KeyT, typename InputT>
class RouterReadableAcceptor : public MultiReadableAcceptor<KeyT, InputT>
{
  public:
    using input_data_t = InputT;

    RouterReadableAcceptor() = default;

    std::shared_ptr<edge::IWritableAcceptor<input_data_t>> get_sink(const KeyT& key) const
    {
        // Simply return an object that will set the message to upstream and go away
        return std::make_shared<UpstreamEdge>(*const_cast<RouterReadableAcceptor<KeyT, InputT>*>(this), key);
    }

    bool has_sink(const KeyT& key) const
    {
        return MultiReadableAcceptor<KeyT, input_data_t>::get_edge_pair(key).first;
    }

    void drop_sink(const KeyT& key)
    {
        MultiReadableAcceptor<KeyT, input_data_t>::release_edge_connection(key);
    }

    void drop_all_sinks()
    {
        MultiReadableAcceptor<KeyT, input_data_t>::release_edge_connections();
    }

  protected:
    class UpstreamEdge : public edge::IReadableAcceptor<input_data_t>
    {
      public:
        UpstreamEdge(edge::IMultiReadableAcceptorBase<KeyT>& parent, KeyT key) : m_parent(parent), m_key(std::move(key))
        {}

        void set_readable_edge_handle(std::shared_ptr<edge::ReadableEdgeHandle> egress) override
        {
            m_parent.set_readable_edge_handle(m_key, std::move(egress));
        }

      private:
        edge::IMultiReadableAcceptorBase<KeyT>& m_parent;
        KeyT m_key;
    };
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class RouterBase : public ForwardingWritableProvider<InputT>, public RouterWritableAcceptor<KeyT, OutputT>
{
  public:
    using input_data_t  = InputT;
    using output_data_t = OutputT;

    RouterBase() : ForwardingWritableProvider<input_data_t>() {}

    // std::shared_ptr<edge::IWritableAcceptor<output_data_t>> get_source(const KeyT& key) const
    // {
    //     // Simply return an object that will set the message to upstream and go away
    //     return std::make_shared<DownstreamEdge>(*const_cast<RouterBase<KeyT, InputT, OutputT>*>(this), key);
    // }

    // bool has_source(const KeyT& key) const
    // {
    //     return MultiSourceProperties<KeyT, output_data_t>::get_edge_pair(key).first;
    // }

    // void drop_edge(const KeyT& key)
    // {
    //     MultiSourceProperties<KeyT, output_data_t>::release_edge_connection(key);
    // }

  protected:
    // class DownstreamEdge : public edge::IWritableAcceptor<output_data_t>
    // {
    //   public:
    //     DownstreamEdge(RouterBase& parent, KeyT key) : m_parent(parent), m_key(std::move(key)) {}

    //     void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    //     {
    //         // Make sure we do any type conversions as needed
    //         auto adapted_ingress = edge::EdgeBuilder::adapt_writable_edge<OutputT>(std::move(ingress));

    //         m_parent.MultiSourceProperties<KeyT, OutputT>::make_edge_connection(m_key, std::move(adapted_ingress));
    //     }

    //   private:
    //     RouterBase<KeyT, input_data_t, output_data_t>& m_parent;
    //     KeyT m_key;
    // };

    void on_complete() override
    {
        MultiSourceProperties<KeyT, output_data_t>::release_edge_connections();
    }
};

template <typename KeyT, typename InputT, typename OutputT = InputT, typename = void>
class Router;

template <typename KeyT, typename InputT, typename OutputT>
class Router<KeyT,
             InputT,
             OutputT,
             std::enable_if_t<!std::is_same_v<InputT, OutputT> && !std::is_convertible_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        KeyT key = this->determine_key_for_value(data);

        auto output = this->convert_value(std::move(data));

        return MultiSourceProperties<KeyT, OutputT>::get_writable_edge(key)->await_write(std::move(output));
    }

    virtual KeyT determine_key_for_value(const InputT& t) = 0;

    virtual OutputT convert_value(InputT&& data) = 0;
};

template <typename KeyT, typename InputT, typename OutputT>
class Router<KeyT,
             InputT,
             OutputT,
             std::enable_if_t<!std::is_same_v<InputT, OutputT> && std::is_convertible_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        KeyT key = this->determine_key_for_value(data);

        return MultiSourceProperties<KeyT, OutputT>::get_writable_edge(key)->await_write(std::move(data));
    }

    virtual KeyT determine_key_for_value(const InputT& t) = 0;
};

template <typename KeyT, typename InputT, typename OutputT>
class Router<KeyT, InputT, OutputT, std::enable_if_t<std::is_same_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        KeyT key = this->determine_key_for_value(data);

        return MultiSourceProperties<KeyT, OutputT>::get_writable_edge(key)->await_write(std::move(data));
    }

    virtual KeyT determine_key_for_value(const InputT& t) = 0;
};

template <typename KeyT, typename T>
class TaggedRouter : public Router<KeyT, std::pair<KeyT, T>, T>
{
  protected:
    using typename RouterBase<KeyT, std::pair<KeyT, T>, T>::input_data_t;
    using typename RouterBase<KeyT, std::pair<KeyT, T>, T>::output_data_t;

    KeyT determine_key_for_value(const input_data_t& data) override
    {
        return data.first;
    }

    output_data_t convert_value(input_data_t&& data) override
    {
        // TODO(MDD): Do we need to move the key too?

        output_data_t tmp = std::move(data.second);
        return tmp;
    }
};

}  // namespace mrc::node
