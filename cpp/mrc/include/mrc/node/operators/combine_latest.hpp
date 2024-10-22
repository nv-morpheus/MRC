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

#include "mrc/channel/status.hpp"
#include "mrc/node/node_parent.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/tuple_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/mutex.hpp>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace mrc::node {

class CombineLatestTypelessBase
{
  public:
    virtual ~CombineLatestTypelessBase() = default;
};

template <typename...>
class CombineLatestBase;

template <typename... InputT, typename OutputT>
class CombineLatestBase<std::tuple<InputT...>, OutputT>
  : public CombineLatestTypelessBase,
    public WritableAcceptor<OutputT>,
    public HeterogeneousNodeParent<edge::IWritableProvider<InputT>...>
{
    template <std::size_t... Is>
    static auto build_ingress(CombineLatestBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

    template <std::size_t... Is>
    static std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...>
    build_child_pairs(CombineLatestBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(
            std::make_pair(MRC_CONCAT_STR("sink[" << Is << "]"), std::ref(*self->get_sink<Is>()))...);
    }

  public:
    using input_tuple_t = std::tuple<InputT...>;
    using output_t      = OutputT;

    CombineLatestBase() :
      m_upstream_holders(build_ingress(const_cast<CombineLatestBase*>(this), std::index_sequence_for<InputT...>{}))
    {}

    ~CombineLatestBase() override = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, InputT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

    std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...> get_children_refs()
        const override
    {
        return build_child_pairs(const_cast<CombineLatestBase*>(this), std::index_sequence_for<InputT...>{});
    }

  protected:
    template <size_t N>
    class Upstream : public ForwardingWritableProvider<NthTypeOf<N, InputT...>>
    {
        using upstream_t = NthTypeOf<N, InputT...>;

      public:
        Upstream(CombineLatestBase& parent) : m_parent(parent) {}

      protected:
        channel::Status on_next(upstream_t&& data) override
        {
            return m_parent.upstream_await_write<N>(std::move(data));
        }

        void on_complete() override
        {
            m_parent.edge_complete();
        }

      private:
        CombineLatestBase& m_parent;
    };

  private:
    template <size_t N>
    channel::Status upstream_await_write(NthTypeOf<N, InputT...> value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the current value
        auto& nth_val = std::get<N>(m_state);

        if (!nth_val.has_value())
        {
            ++m_values_set;
        }

        nth_val = std::move(value);

        channel::Status status = channel::Status::success;

        // Check if we should push the new value
        if (m_values_set == sizeof...(InputT))
        {
            std::tuple<InputT...> new_val = utils::tuple_surely(m_state);

            status = this->get_writable_edge()->await_write(this->convert_value(std::move(new_val)));
        }

        return status;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(InputT))
        {
            // Clear the held tuple to remove any dangling values
            m_state = std::tuple<std::optional<InputT>...>();

            WritableAcceptor<OutputT>::release_edge_connection();
        }
    }

    virtual output_t convert_value(input_tuple_t&& data) = 0;

    mutable Mutex m_mutex;

    // The number of elements that have been set. Can start emitting when m_values_set == sizeof...(TypesT)
    size_t m_values_set{0};

    // Counts the number of upstream completions. When m_completions == sizeof...(TypesT), the downstream edges are
    // released
    size_t m_completions{0};

    // Holds onto the latest values to eventually push when new ones are emitted
    std::tuple<std::optional<InputT>...> m_state;

    // Upstream edges
    std::tuple<std::shared_ptr<WritableProvider<InputT>>...> m_upstream_holders;
};

template <typename...>
class CombineLatestComponent;

template <typename... InputT, typename OutputT>
class CombineLatestComponent<std::tuple<InputT...>, OutputT> : public CombineLatestBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t        = CombineLatestBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;
};

// Specialization for CombineLatest with a default output type
template <typename... InputT>
class CombineLatestComponent<std::tuple<InputT...>>
  : public CombineLatestBase<std::tuple<InputT...>, std::tuple<InputT...>>
{
  public:
    using base_t        = CombineLatestBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;

  private:
    output_t convert_value(input_tuple_t&& data) override
    {
        // No change to the output type
        return std::move(data);
    }
};

template <typename...>
class CombineLatestTransformComponent;

template <typename... InputT, typename OutputT>
class CombineLatestTransformComponent<std::tuple<InputT...>, OutputT>
  : public CombineLatestBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t         = CombineLatestBase<std::tuple<InputT...>, OutputT>;
    using input_tuple_t  = typename base_t::input_tuple_t;
    using output_t       = typename base_t::output_t;
    using transform_fn_t = std::function<output_t(input_tuple_t&&)>;

    CombineLatestTransformComponent(transform_fn_t transform_fn) : base_t(), m_transform_fn(std::move(transform_fn)) {}

  private:
    output_t convert_value(input_tuple_t&& data) override
    {
        return m_transform_fn(std::move(data));
    }

    transform_fn_t m_transform_fn;
};

}  // namespace mrc::node
