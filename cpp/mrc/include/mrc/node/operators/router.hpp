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

#include "mrc/channel/status.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/operators/operator.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/node/source_properties.hpp"

#include <map>
#include <memory>
#include <type_traits>

namespace mrc::node {

// template <typename KeyT, typename T>
// class RouterBase
// {
//     std::map<KeyT, SourceChannelWriteable<T>> m_sources;

//   protected:
//     inline SourceChannelWriteable<T>& channel_for_key(const KeyT& key)
//     {
//         auto search = m_sources.find(key);
//         if (search == m_sources.end())
//         {
//             throw exceptions::MrcRuntimeError("unable to find edge for key");
//         }
//         return search->second;
//     }

//     void release_sources()
//     {
//         m_sources.clear();
//     }

//   public:
//     SourceChannel<T>& source(KeyT key)
//     {
//         return m_sources[key];
//     }

//     bool has_edge(KeyT key) const
//     {
//         auto search = m_sources.find(key);
//         return (search != m_sources.end());
//     }

//     void drop_edge(KeyT key)
//     {
//         auto search = m_sources.find(key);
//         if (search != m_sources.end())
//         {
//             m_sources.erase(search);
//         }
//     }
// };

// template <typename KeyT, typename T>
// class Router : public Operator<std::pair<KeyT, T>>, public RouterBase<KeyT, T>
// {
//     // Operator::on_next
//     inline channel::Status on_next(std::pair<KeyT, T>&& tagged_data) final
//     {
//         return this->channel_for_key(tagged_data.first).await_write(std::move(tagged_data.second));
//     }

//     // Operator::on_complete
//     void on_complete() final
//     {
//         this->release_sources();
//     }
// };

template <typename KeyT, typename InputT, typename OutputT = InputT>
class RouterBase : public ForwardingIngressProvider<InputT>, public MultiSourceProperties<OutputT, KeyT>
{
  public:
    using input_data_t  = InputT;
    using output_data_t = OutputT;

    RouterBase() : ForwardingIngressProvider<input_data_t>() {}

    std::shared_ptr<IIngressAcceptor<output_data_t>> get_source(const KeyT& key) const
    {
        // Simply return an object that will set the message to upstream and go away
        return std::make_shared<DownstreamEdge>(*const_cast<RouterBase<KeyT, InputT, OutputT>*>(this), key);
    }

    bool has_source(const KeyT& key) const
    {
        return MultiSourceProperties<output_data_t, KeyT>::get_edge_pair(key).first;
    }

    void drop_edge(const KeyT& key)
    {
        MultiSourceProperties<output_data_t, KeyT>::release_edge_connection(key);
    }

  protected:
    class DownstreamEdge : public IIngressAcceptor<output_data_t>
    {
      public:
        DownstreamEdge(RouterBase& parent, KeyT key) : m_parent(parent), m_key(std::move(key)) {}

        void set_ingress_obj(std::shared_ptr<IngressHandleObj> ingress) override
        {
            // Make sure we do any type conversions as needed
            auto adapted_ingress = EdgeBuilder::adapt_ingress<OutputT>(std::move(ingress));

            m_parent.MultiSourceProperties<OutputT, KeyT>::make_edge_connection(m_key, std::move(adapted_ingress));
        }

      private:
        RouterBase<KeyT, input_data_t, output_data_t>& m_parent;
        KeyT m_key;
    };

    // channel::Status on_next(input_data_t&& data) override
    // {
    //     KeyT key = this->determine_key_for_value(data);

    //     auto output = this->convert_value(std::move(data));

    //     return MultiSourceProperties<output_data_t, KeyT>::get_writable_edge(key)->await_write(std::move(output));
    // }

    void on_complete() override
    {
        MultiSourceProperties<output_data_t, KeyT>::release_edge_connections();
    }

    // virtual KeyT determine_key_for_value(const input_data_t& t) = 0;

    // virtual output_data_t convert_value(input_data_t&& t) = 0;
};

// template <typename KeyT, typename T>
// class RouterBase<KeyT, T, T> : public ForwardingIngressProvider<T>, public MultiSourceProperties<T, KeyT>
// {
//   public:
//     using input_data_t  = T;
//     using output_data_t = T;

//     RouterBase() : ForwardingIngressProvider<input_data_t>() {}

//     std::shared_ptr<IIngressAcceptor<input_data_t>> get_source(const KeyT& key) const
//     {
//         // Simply return an object that will set the message to upstream and go away
//         return std::make_shared<DownstreamEdge>(*this, key);
//     }

//     bool has_source(const KeyT& key) const
//     {
//         return MultiSourceProperties<output_data_t, KeyT>::get_edge_pair(key).first;
//     }

//     void drop_edge(const KeyT& key)
//     {
//         MultiSourceProperties<output_data_t, KeyT>::release_edge(key);
//     }

//   protected:
//     class DownstreamEdge : public IIngressAcceptor<output_data_t>
//     {
//       public:
//         DownstreamEdge(RouterBase& parent, KeyT key) : m_parent(parent), m_key(std::move(key)) {}

//         void set_ingress(std::shared_ptr<EdgeWritable<output_data_t>> ingress) override
//         {
//             m_parent.set_edge(m_key, std::move(ingress));
//         }

//       private:
//         RouterBase<KeyT, input_data_t, output_data_t> m_parent;
//         KeyT m_key;
//     };

//     channel::Status on_next(input_data_t&& data) override
//     {
//         KeyT key = this->determine_key_for_value(data);

//         return MultiSourceProperties<output_data_t, KeyT>::get_writable_edge(key)->await_write(std::move(data));
//     }

//     virtual void on_complete()
//     {
//         MultiSourceProperties<output_data_t, KeyT>::release_edges();
//     }

//     virtual KeyT determine_key_for_value(const input_data_t& t) = 0;
// };

template <typename KeyT, typename InputT, typename OutputT = InputT, typename = void>
class Router : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        KeyT key = this->determine_key_for_value(data);

        auto output = this->convert_value(std::move(data));

        return MultiSourceProperties<OutputT, KeyT>::get_writable_edge(key)->await_write(std::move(output));
    }

    virtual KeyT determine_key_for_value(const InputT& t) = 0;

    virtual OutputT convert_value(InputT&& data) = 0;
};

template <typename KeyT, typename InputT, typename OutputT>
class Router<KeyT, InputT, OutputT, std::enable_if_t<std::is_convertible_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        KeyT key = this->determine_key_for_value(data);

        return MultiSourceProperties<OutputT, KeyT>::get_writable_edge(key)->await_write(std::move(data));
    }

    virtual KeyT determine_key_for_value(const InputT& t) = 0;

    OutputT convert_value(InputT&& data)
    {
        return data;
    }
};

// template <typename KeyT, typename InputT, typename OutputT = InputT>
// class Router : public RouterBase<KeyT, InputT, OutputT>
// {
//   protected:
//     channel::Status on_next(InputT&& data) override
//     {
//         KeyT key = this->determine_key_for_value(data);

//         if constexpr (std::is_convertible_v<InputT, OutputT>)
//         {
//             return MultiSourceProperties<OutputT, KeyT>::get_writable_edge(key)->await_write(std::move(data));
//         }
//         else
//         {
//             // If not convertable, call convert_value
//             return MultiSourceProperties<OutputT, KeyT>::get_writable_edge(key)->await_write(
//                 this->convert_value(std::move(data)));
//         }
//     }

//     // If the value is convertable, implement the abstract method
//     std::enable_if_t<!std::is_convertible_v<InputT, OutputT>, OutputT> convert_value(InputT&& data) override
//     {
//         return data;
//     }
// };

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
