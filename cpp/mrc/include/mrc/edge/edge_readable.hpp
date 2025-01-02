/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/edge/edge.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>
#include <sys/types.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc::edge {

class IEdgeReadableBase : public virtual EdgeBase
{
  public:
    ~IEdgeReadableBase() override = default;

    virtual EdgeTypeInfo get_type() const = 0;
};

template <typename T>
class IEdgeReadable : public virtual Edge<T>, public IEdgeReadableBase
{
  public:
    EdgeTypeInfo get_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }

    virtual channel::Status await_read(T& t) = 0;
};

template <typename InputT, typename OutputT = InputT>
class ConvertingEdgeReadableBase : public IEdgeReadable<OutputT>
{
  public:
    using input_t  = InputT;
    using output_t = OutputT;

    ConvertingEdgeReadableBase(std::shared_ptr<IEdgeReadable<InputT>> upstream) : m_upstream(upstream)
    {
        this->add_linked_edge(upstream);
    }

  protected:
    inline IEdgeReadable<InputT>& upstream() const
    {
        return *m_upstream;
    }

  private:
    std::shared_ptr<IEdgeReadable<InputT>> m_upstream{};
};

template <typename InputT, typename OutputT = InputT, typename EnableT = void>
class ConvertingEdgeReadable;

template <typename InputT, typename OutputT>
class ConvertingEdgeReadable<InputT, OutputT, std::enable_if_t<std::is_convertible_v<InputT, OutputT>>>
  : public ConvertingEdgeReadableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeReadableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;

    using base_t::base_t;

    channel::Status await_read(OutputT& data) override
    {
        InputT source_data;
        auto ret_val = this->upstream().await_read(source_data);

        // Convert to the sink type
        data = std::move(source_data);

        return ret_val;
    }
};

template <typename InputT, typename OutputT>
class LambdaConvertingEdgeReadable : public ConvertingEdgeReadableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeReadableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;
    using lambda_fn_t = std::function<output_t(input_t&&)>;

    LambdaConvertingEdgeReadable(lambda_fn_t lambda_fn, std::shared_ptr<IEdgeReadable<input_t>> upstream) :
      ConvertingEdgeReadableBase<input_t, output_t>(upstream),
      m_lambda_fn(std::move(lambda_fn))
    {}

    channel::Status await_read(output_t& data) override
    {
        input_t source_data;
        auto ret_val = this->upstream().await_read(source_data);

        // Convert to the sink type
        data = m_lambda_fn(std::move(source_data));

        return ret_val;
    }

  private:
    lambda_fn_t m_lambda_fn{};
};

class ReadableEdgeHandle : public EdgeHandle
{
  public:
    ReadableEdgeHandle(std::shared_ptr<IEdgeReadableBase> egress) : EdgeHandle(egress->get_type(), egress) {}

    static std::shared_ptr<ReadableEdgeHandle> from_typeless(std::shared_ptr<EdgeHandle> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeReadableBase>();

        CHECK(typed_ingress) << "Could not convert to egress";

        return std::make_shared<ReadableEdgeHandle>(std::move(typed_ingress));
    }

  private:
    std::shared_ptr<IEdgeReadableBase> get_egress() const
    {
        return std::dynamic_pointer_cast<IEdgeReadableBase>(this->get_handle());
    }

    template <typename T>
    std::shared_ptr<IEdgeReadable<T>> get_egress_typed() const
    {
        return std::dynamic_pointer_cast<IEdgeReadable<T>>(this->get_handle());
    }

    void set_egress_handle(std::shared_ptr<IEdgeReadableBase> egress)
    {
        this->m_type   = egress->get_type();
        this->m_handle = egress;
    }

    friend EdgeBuilder;
};

class IReadableProviderBase
{
  public:
    virtual std::shared_ptr<ReadableEdgeHandle> get_readable_edge_handle() const = 0;

    virtual EdgeTypeInfo readable_provider_type() const = 0;
};

class IReadableAcceptorBase
{
  public:
    virtual void set_readable_edge_handle(std::shared_ptr<ReadableEdgeHandle> egress) = 0;

    virtual EdgeTypeInfo readable_acceptor_type() const = 0;
};

template <typename KeyT>
class IMultiReadableProviderBase
{
  public:
    virtual bool has_readable_edge(const KeyT& key) const                                = 0;
    virtual void release_readable_edge(const KeyT& key)                                  = 0;
    virtual void release_readable_edges()                                                = 0;
    virtual size_t readable_edge_count() const                                           = 0;
    virtual std::vector<KeyT> readable_edge_keys() const                                 = 0;
    virtual std::shared_ptr<ReadableEdgeHandle> get_readable_edge_handle(KeyT key) const = 0;
};

template <typename KeyT>
class IMultiReadableAcceptorBase
{
  public:
    virtual bool has_readable_edge(const KeyT& key) const                                       = 0;
    virtual void release_readable_edge(const KeyT& key)                                         = 0;
    virtual void release_readable_edges()                                                       = 0;
    virtual size_t readable_edge_count() const                                                  = 0;
    virtual std::vector<KeyT> readable_edge_keys() const                                        = 0;
    virtual void set_readable_edge_handle(KeyT key, std::shared_ptr<ReadableEdgeHandle> egress) = 0;
};

template <typename T>
class IReadableProvider : public virtual IReadableProviderBase
{
  public:
    EdgeTypeInfo readable_provider_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }
};

template <typename T>
class IReadableAcceptor : public virtual IReadableAcceptorBase
{
  public:
    EdgeTypeInfo readable_acceptor_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }
};

template <typename KeyT, typename T>
class IMultiReadableProvider : public virtual IMultiReadableProviderBase<KeyT>
{};

template <typename KeyT, typename T>
class IMultiReadableAcceptor : public virtual IMultiReadableAcceptorBase<KeyT>
{};
}  // namespace mrc::edge
