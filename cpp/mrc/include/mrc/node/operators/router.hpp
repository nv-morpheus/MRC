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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/node_parent.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/condition_variable.hpp>

#include <map>
#include <memory>
#include <queue>
#include <stop_token>
#include <type_traits>

// IWYU pragma: begin_exports

namespace mrc::node {

template <typename InputT>
class RouterDownstreamNode : public edge::IWritableAcceptor<InputT>,
                             public edge::IReadableProvider<InputT>,
                             public ISourceChannelOwner<InputT>
{};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class RouterBase : public MultiWritableAcceptor<KeyT, OutputT>,
                   public MultiReadableProvider<KeyT, OutputT>,
                   public MultiSourceChannelOwner<KeyT, OutputT>
{
  public:
    virtual std::shared_ptr<RouterDownstreamNode<OutputT>> get_source(const KeyT& key) const = 0;

    bool has_source(const KeyT& key) const
    {
        return MultiSourceProperties<KeyT, OutputT>::get_edge_pair(key).first;
    }

    void drop_source(const KeyT& key)
    {
        MultiSourceProperties<KeyT, OutputT>::release_edge_connection(key);
    }

  protected:
    class Downstream : public RouterDownstreamNode<OutputT>
    {
      public:
        Downstream(RouterBase& parent, KeyT key) : m_parent(parent), m_key(std::move(key))
        {
            this->set_channel(std::make_unique<mrc::channel::BufferedChannel<OutputT>>());
        }

        void set_channel(std::unique_ptr<mrc::channel::Channel<OutputT>> channel) override
        {
            m_parent.MultiSourceChannelOwner<KeyT, OutputT>::set_channel(m_key, std::move(channel));
        }

        void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
        {
            m_parent.MultiWritableAcceptor<KeyT, OutputT>::set_writable_edge_handle(m_key, std::move(ingress));
        }

        std::shared_ptr<edge::ReadableEdgeHandle> get_readable_edge_handle() const override
        {
            return m_parent.MultiReadableProvider<KeyT, OutputT>::get_readable_edge_handle(m_key);
        }

      private:
        RouterBase& m_parent;
        KeyT m_key;
    };

    virtual KeyT determine_key_for_value(const InputT& t) = 0;

    virtual OutputT convert_value(InputT&& data) = 0;

    channel::Status process_one(InputT&& data)
    {
        try
        {
            KeyT key = this->determine_key_for_value(data);

            if constexpr (std::is_same_v<InputT, OutputT> || std::is_convertible_v<InputT, OutputT>)
            {
                return MultiSourceProperties<KeyT, OutputT>::get_writable_edge(key)->await_write(std::move(data));
            }
            else
            {
                OutputT output = this->convert_value(std::move(data));

                return MultiSourceProperties<KeyT, OutputT>::get_writable_edge(key)->await_write(std::move(output));
            }

        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Caught exception: " << e.what() << std::endl;
            return channel::Status::error;
        }
    }
};

template <typename KeyT, typename InputT, typename OutputT = InputT, typename = void>
class ConvertingRouterBase;

template <typename KeyT, typename InputT, typename OutputT>
class ConvertingRouterBase<KeyT,
                           InputT,
                           OutputT,
                           std::enable_if_t<!std::is_same_v<InputT, OutputT> && !std::is_convertible_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{};

template <typename KeyT, typename InputT, typename OutputT>
class ConvertingRouterBase<KeyT,
                           InputT,
                           OutputT,
                           std::enable_if_t<std::is_same_v<InputT, OutputT> || std::is_convertible_v<InputT, OutputT>>>
  : public RouterBase<KeyT, InputT, OutputT>
{
  protected:
    OutputT convert_value(InputT&& data) override
    {
        // This is a no-op, we just return the data. This wont be used.
        return std::move(data);
    }
};

template <typename KeyT, typename InputT, typename OutputT = InputT, typename = void>
class LambdaRouterBase;

template <typename KeyT, typename InputT, typename OutputT>
class LambdaRouterBase<KeyT,
                       InputT,
                       OutputT,
                       std::enable_if_t<!std::is_same_v<InputT, OutputT> && !std::is_convertible_v<InputT, OutputT>>>
  : public virtual ConvertingRouterBase<KeyT, InputT, OutputT>
{
  public:
    using base_t       = ConvertingRouterBase<KeyT, InputT, OutputT>;
    using key_fn_t     = std::function<KeyT(const InputT&)>;
    using convert_fn_t = std::function<OutputT(InputT&&)>;

    LambdaRouterBase(key_fn_t key_fn, convert_fn_t convert_fn) :
      base_t(),
      m_key_fn(std::move(key_fn)),
      m_convert_fn(std::move(convert_fn))
    {}

  protected:
    KeyT determine_key_for_value(const InputT& t) override
    {
        return m_key_fn(t);
    }

    OutputT convert_value(InputT&& data) override
    {
        return m_convert_fn(std::move(data));
    }

    key_fn_t m_key_fn;
    convert_fn_t m_convert_fn;
};

template <typename KeyT, typename InputT, typename OutputT>
class LambdaRouterBase<KeyT,
                       InputT,
                       OutputT,
                       std::enable_if_t<std::is_same_v<InputT, OutputT> || std::is_convertible_v<InputT, OutputT>>>
  : public virtual ConvertingRouterBase<KeyT, InputT, OutputT>
{
  public:
    using base_t   = ConvertingRouterBase<KeyT, InputT, OutputT>;
    using key_fn_t = std::function<KeyT(const InputT&)>;

    LambdaRouterBase(key_fn_t key_fn) : base_t(), m_key_fn(std::move(key_fn)) {}

  protected:
    KeyT determine_key_for_value(const InputT& t) override
    {
        return m_key_fn(t);
    }

    key_fn_t m_key_fn;
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class StaticRouterBase : public virtual ConvertingRouterBase<KeyT, InputT, OutputT>,
                         public HomogeneousNodeParent<RouterDownstreamNode<OutputT>>
{
  public:
    using base_t = ConvertingRouterBase<KeyT, InputT, OutputT>;
    using this_t = StaticRouterBase<KeyT, InputT, OutputT>;

    StaticRouterBase(std::vector<KeyT> route_keys)
    {
        // Create a downstream for each key
        for (const auto& key : route_keys)
        {
            m_downstreams[key] = std::make_shared<typename this_t::Downstream>(*this, key);
        }
    }

    std::shared_ptr<RouterDownstreamNode<OutputT>> get_source(const KeyT& key) const override
    {
        if (!m_downstreams.contains(key))
        {
            throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Key '" << key << "' found in router"));
        }

        return m_downstreams.at(key);
    }

    std::map<std::string, std::reference_wrapper<typename this_t::child_node_t>> get_children_refs(
        std::optional<std::string> child_name = std::nullopt) const override
    {
        std::map<std::string, std::reference_wrapper<typename this_t::child_node_t>> children;

        for (const auto& [key, downstream] : m_downstreams)
        {
            // Utilize MRC_CONCAT_STR to convert the type to a string as best we can
            children.emplace(MRC_CONCAT_STR(key), std::ref(*downstream));
        }

        return children;
    }

  protected:
    std::map<KeyT, std::shared_ptr<typename this_t::Downstream>> m_downstreams;
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class DynamicRouterBase : public virtual ConvertingRouterBase<KeyT, InputT, OutputT>
{
    using this_t = DynamicRouterBase<KeyT, InputT, OutputT>;

  public:
    std::shared_ptr<RouterDownstreamNode<OutputT>> get_source(const KeyT& key) const override
    {
        std::shared_ptr<typename this_t::Downstream> downstream;

        if (!m_downstreams.contains(key) || (downstream = m_downstreams.at(key).lock()) == nullptr)
        {
            // Cast away constness to create the downstream
            auto non_const_this = const_cast<this_t*>(this);

            downstream = std::make_shared<typename this_t::Downstream>(*non_const_this, key);

            non_const_this->m_downstreams[key] = downstream;

            return downstream;
        }

        return downstream;
    }

  protected:
    std::map<KeyT, std::weak_ptr<typename this_t::Downstream>> m_downstreams;
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class ComponentRouterBase : public ForwardingWritableProvider<InputT>,
                            public virtual ConvertingRouterBase<KeyT, InputT, OutputT>
{
  protected:
    channel::Status on_next(InputT&& data) override
    {
        return this->process_one(std::move(data));
    }

    void on_complete() override
    {
        MultiSourceProperties<KeyT, OutputT>::release_edge_connections();
    }
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class RunnableRouterBase : public WritableProvider<InputT>,
                           public ReadableAcceptor<InputT>,
                           public SinkChannelOwner<InputT>,
                           public virtual ConvertingRouterBase<KeyT, InputT, OutputT>,
                           public mrc::runnable::RunnableWithContext<>
{
  protected:
    RunnableRouterBase()
    {
        SinkChannelOwner<InputT>::set_channel(std::make_unique<mrc::channel::BufferedChannel<InputT>>());
    }

    // Allows for easier testing of this method
    void do_run()
    {
        InputT data;
        channel::Status read_status;
        channel::Status write_status = channel::Status::success;  // give an initial value

        // Loop until either the node has been killed or the upstream terminated
        while (!m_stop_source.stop_requested() &&
               (read_status = this->get_readable_edge()->await_read(data)) == channel::Status::success &&
               write_status == channel::Status::success)
        {
            write_status = this->process_one(std::move(data));
        }

        // Drop all connections

        if (read_status == channel::Status::error)
        {
            throw exceptions::MrcRuntimeError("Failed to read from upstream");
        }

        if (write_status == channel::Status::error)
        {
            throw exceptions::MrcRuntimeError("Failed to write to downstream");
        }
    }

  private:
    /**
     * @brief Runnable's entrypoint.
     */
    void run(mrc::runnable::Context& ctx) override
    {
        Unwinder unwinder([&] {
            ctx.barrier();

            if (ctx.rank() == 0)
            {
                MultiSourceProperties<KeyT, OutputT>::release_edge_connections();
            }
        });

        this->do_run();
    }

    /**
     * @brief Runnable's state control, for stopping from MRC.
     */
    void on_state_update(const mrc::runnable::Runnable::State& state) final
    {
        switch (state)
        {
        case mrc::runnable::Runnable::State::Stop:
            // Do nothing, we wait for the upstream channel to return closed
            // m_stop_source.request_stop();
            break;

        case mrc::runnable::Runnable::State::Kill:
            m_stop_source.request_stop();
            break;

        default:
            break;
        }
    }

    std::stop_source m_stop_source;
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class StaticRouterComponentBase : public StaticRouterBase<KeyT, InputT, OutputT>,
                                  public ComponentRouterBase<KeyT, InputT, OutputT>
{
  public:
    StaticRouterComponentBase(std::vector<KeyT> route_keys) :
      StaticRouterBase<KeyT, InputT, OutputT>(std::move(route_keys))
    {}
};

template <typename KeyT, typename InputT, typename OutputT = InputT, typename = void>
class LambdaStaticRouterComponent;

template <typename KeyT, typename InputT, typename OutputT>
class LambdaStaticRouterComponent<
    KeyT,
    InputT,
    OutputT,
    std::enable_if_t<!std::is_same_v<InputT, OutputT> && !std::is_convertible_v<InputT, OutputT>>>
  : public LambdaRouterBase<KeyT, InputT, OutputT>, public StaticRouterComponentBase<KeyT, InputT, OutputT>
{
  public:
    using key_fn_t     = LambdaRouterBase<KeyT, InputT, OutputT>::key_fn_t;
    using convert_fn_t = LambdaRouterBase<KeyT, InputT, OutputT>::convert_fn_t;

    LambdaStaticRouterComponent(std::vector<KeyT> route_keys, key_fn_t key_fn, convert_fn_t convert_fn) :
      LambdaRouterBase<KeyT, InputT, OutputT>(std::move(key_fn), std::move(convert_fn)),
      StaticRouterComponentBase<KeyT, InputT, OutputT>(std::move(route_keys))
    {}
};

template <typename KeyT, typename InputT, typename OutputT>
class LambdaStaticRouterComponent<
    KeyT,
    InputT,
    OutputT,
    std::enable_if_t<std::is_same_v<InputT, OutputT> || std::is_convertible_v<InputT, OutputT>>>
  : public LambdaRouterBase<KeyT, InputT, OutputT>, public StaticRouterComponentBase<KeyT, InputT, OutputT>
{
  public:
    using key_fn_t = LambdaRouterBase<KeyT, InputT, OutputT>::key_fn_t;

    LambdaStaticRouterComponent(std::vector<KeyT> route_keys, key_fn_t key_fn) :
      LambdaRouterBase<KeyT, InputT, OutputT>(std::move(key_fn)),
      StaticRouterComponentBase<KeyT, InputT, OutputT>(std::move(route_keys))
    {}
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class StaticRouterRunnableBase : public StaticRouterBase<KeyT, InputT, OutputT>,
                                 public RunnableRouterBase<KeyT, InputT, OutputT>
{
  public:
    StaticRouterRunnableBase(std::vector<KeyT> route_keys) :
      StaticRouterBase<KeyT, InputT, OutputT>(std::move(route_keys))
    {}
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class LambdaStaticRouterRunnable : public LambdaRouterBase<KeyT, InputT, OutputT>,
                                   public StaticRouterRunnableBase<KeyT, InputT, OutputT>
{
  public:
    using key_fn_t = std::function<KeyT(const InputT&)>;

    LambdaStaticRouterRunnable(std::vector<KeyT> route_keys, key_fn_t key_fn) :
      StaticRouterRunnableBase<KeyT, InputT, OutputT>(std::move(route_keys)),
      LambdaRouterBase<KeyT, InputT, OutputT>(std::move(key_fn))
    {}
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class DynamicRouterComponentBase : public DynamicRouterBase<KeyT, InputT, OutputT>,
                                   public ComponentRouterBase<KeyT, InputT, OutputT>
{};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class LambdaDynamicRouterComponent : public LambdaRouterBase<KeyT, InputT, OutputT>,
                                     public DynamicRouterComponentBase<KeyT, InputT, OutputT>
{
  public:
    using LambdaRouterBase<KeyT, InputT, OutputT>::LambdaRouterBase;
};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class DynamicRouterRunnableBase : public DynamicRouterBase<KeyT, InputT, OutputT>,
                                  public RunnableRouterBase<KeyT, InputT, OutputT>
{};

template <typename KeyT, typename InputT, typename OutputT = InputT>
class LambdaDynamicRouterRunnable : public LambdaRouterBase<KeyT, InputT, OutputT>,
                                    public DynamicRouterRunnableBase<KeyT, InputT, OutputT>
{
  public:
    using LambdaRouterBase<KeyT, InputT, OutputT>::LambdaRouterBase;
};

template <typename KeyT, typename T>
class TaggedRouter : public DynamicRouterComponentBase<KeyT, std::pair<KeyT, T>, T>
{
  protected:
    KeyT determine_key_for_value(const std::pair<KeyT, T>& data) override
    {
        return data.first;
    }

    T convert_value(std::pair<KeyT, T>&& data) override
    {
        // TODO(MDD): Do we need to move the key too?

        T tmp = std::move(data.second);
        return tmp;
    }
};

}  // namespace mrc::node

// IWYU pragma: end_exports
