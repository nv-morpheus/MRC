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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/edge/edge.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/exceptions/runtime_error.hpp"
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

class IEdgeWritableBase : public virtual EdgeBase
{
  public:
    ~IEdgeWritableBase() override = default;

    virtual EdgeTypeInfo get_type() const = 0;
};

template <typename T>
class IEdgeWritable : public virtual Edge<T>, public virtual IEdgeWritableBase
{
  public:
    EdgeTypeInfo get_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }

    virtual channel::Status await_write(T&& data) = 0;

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline channel::Status await_write(T data)
    {
        return await_write(std::move(data));
    }
};

template <typename InputT, typename OutputT = InputT>
class ConvertingEdgeWritableBase : public IEdgeWritable<InputT>
{
  public:
    using input_t  = InputT;
    using output_t = OutputT;

    ConvertingEdgeWritableBase(std::shared_ptr<IEdgeWritable<OutputT>> downstream) : m_downstream(downstream)
    {
        this->add_linked_edge(downstream);
    }

  protected:
    inline IEdgeWritable<OutputT>& downstream() const
    {
        return *m_downstream;
    }

  private:
    std::shared_ptr<IEdgeWritable<OutputT>> m_downstream{};
};

template <typename InputT, typename OutputT = InputT, typename EnableT = void>
class ConvertingEdgeWritable;

template <typename InputT, typename OutputT>
class ConvertingEdgeWritable<InputT, OutputT, std::enable_if_t<std::is_convertible_v<InputT, OutputT>>>
  : public ConvertingEdgeWritableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;

    using base_t::base_t;

    channel::Status await_write(input_t&& data) override
    {
        return this->downstream().await_write(std::move(data));
    }
};

template <typename InputT, typename OutputT>
class LambdaConvertingEdgeWritable : public ConvertingEdgeWritableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;
    using lambda_fn_t = std::function<output_t(input_t&&)>;

    LambdaConvertingEdgeWritable(lambda_fn_t lambda_fn, std::shared_ptr<IEdgeWritable<output_t>> downstream) :
      ConvertingEdgeWritableBase<input_t, output_t>(downstream),
      m_lambda_fn(std::move(lambda_fn))
    {}

    channel::Status await_write(input_t&& data) override
    {
        return this->downstream().await_write(m_lambda_fn(std::move(data)));
    }

  private:
    lambda_fn_t m_lambda_fn{};
};

class WritableEdgeHandle : public EdgeHandle
{
  public:
    WritableEdgeHandle(std::shared_ptr<IEdgeWritableBase> ingress) : WritableEdgeHandle(ingress->get_type(), ingress) {}

    static std::shared_ptr<WritableEdgeHandle> from_typeless(std::shared_ptr<EdgeHandle> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeWritableBase>();

        CHECK(typed_ingress) << "Could not convert to ingress";

        return std::make_shared<WritableEdgeHandle>(std::move(typed_ingress));
    }

    virtual bool is_deferred() const
    {
        return false;
    }

  protected:
    // Allow manually specifying the edge type
    WritableEdgeHandle(EdgeTypeInfo edge_type, std::shared_ptr<IEdgeWritableBase> ingress) :
      EdgeHandle(edge_type, ingress)
    {}

  private:
    std::shared_ptr<IEdgeWritableBase> get_ingress() const
    {
        return std::dynamic_pointer_cast<IEdgeWritableBase>(this->get_handle());
    }

    template <typename T>
    std::shared_ptr<IEdgeWritable<T>> get_ingress_typed() const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_handle());
    }

    void set_ingress_handle(std::shared_ptr<IEdgeWritableBase> ingress)
    {
        this->m_type   = ingress->get_type();
        this->m_handle = ingress;
    }

    // Allow EdgeBuilder to unpack the edge
    friend EdgeBuilder;

    // Add deferred ingresses to set their deferred type
    friend DeferredWritableEdgeHandle;
};

class IWritableProviderBase
{
  public:
    virtual std::shared_ptr<WritableEdgeHandle> get_writable_edge_handle() const = 0;

    virtual EdgeTypeInfo writable_provider_type() const = 0;
};

class IWritableAcceptorBase
{
  public:
    virtual void set_writable_edge_handle(std::shared_ptr<WritableEdgeHandle> ingress) = 0;

    virtual EdgeTypeInfo writable_acceptor_type() const = 0;
};

template <typename KeyT>
class IMultiWritableAcceptorBase
{
  public:
    virtual bool has_writable_edge(const KeyT& key) const = 0;
    virtual void release_writable_edge(const KeyT& key)   = 0;
    virtual void release_writable_edges()                 = 0;
    virtual size_t writable_edge_count() const            = 0;
    virtual std::vector<KeyT> writable_edge_keys() const  = 0;

    virtual void set_writable_edge_handle(KeyT key, std::shared_ptr<WritableEdgeHandle> ingress) = 0;
};

template <typename T>
class IWritableProvider : public virtual IWritableProviderBase
{
  public:
    EdgeTypeInfo writable_provider_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }
};

template <typename T>
class IWritableAcceptor : public virtual IWritableAcceptorBase
{
  public:
    EdgeTypeInfo writable_acceptor_type() const override
    {
        return EdgeTypeInfo::create<T>();
    }
};

template <typename KeyT, typename T>
class IMultiWritableAcceptor : public virtual IMultiWritableAcceptorBase<KeyT>
{};

}  // namespace mrc::edge
