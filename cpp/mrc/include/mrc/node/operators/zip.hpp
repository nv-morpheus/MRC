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
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/mutex.hpp>

#include <execution>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>

namespace mrc::node {

template <typename... TypesT>
class ZipBase
{};

template <typename... InputT, typename OutputT>
class ZipBase<std::tuple<InputT...>, OutputT> : public WritableAcceptor<OutputT>
{
    template <std::size_t... Is>
    static auto build_ingress(ZipBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

  public:
    using input_tuple_t = std::tuple<InputT...>;
    using output_t      = OutputT;

    ZipBase(size_t max_outstanding = 64) :
      m_upstream_holders(build_ingress(const_cast<ZipBase*>(this), std::index_sequence_for<InputT...>{})),
      m_max_outstanding(max_outstanding)
    {}

    virtual ~ZipBase() = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, InputT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

  protected:
    template <size_t N>
    class Upstream : public WritableProvider<NthTypeOf<N, InputT...>>
    {
        using upstream_t = NthTypeOf<N, InputT...>;

      public:
        Upstream(ZipBase& parent)
        {
            this->init_owned_edge(std::make_shared<InnerEdge>(parent));
        }

      private:
        class InnerEdge : public edge::IEdgeWritable<NthTypeOf<N, InputT...>>
        {
          public:
            InnerEdge(ZipBase& parent) : m_parent(parent) {}
            ~InnerEdge()
            {
                m_parent.edge_complete();
            }

            channel::Status await_write(upstream_t&& data) override
            {
                return m_parent.set_upstream_value<N>(std::move(data));
            }

          private:
            ZipBase& m_parent;
        };
    };

  private:
    template <size_t N>
    channel::Status set_upstream_value(NthTypeOf<N, InputT...> value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the current value
        auto& nth_vector = std::get<N>(m_state);

        // Ensure we have room, otherwise block
        m_cv.wait(lock, [this, &nth_vector] {
            return nth_vector.size() < m_max_outstanding;
        });

        // Push the new value
        nth_vector.push(std::move(value));

        channel::Status status = channel::Status::success;

        bool all_have_values = std::apply(
            [](auto const&... vec) {
                return (!vec.empty() && ...);
            },
            m_state);

        // Check if we should push the new value
        if (all_have_values)
        {
            // Pop the front of each vector and push the tuple
            input_tuple_t new_val = std::apply(
                [](auto&... vec) {
                    // Move the front of each vector into a tuple
                    auto tmp_tuple = std::make_tuple(std::move(vec.front())...);

                    // Pop the front of each vector
                    (vec.pop(), ...);

                    // Return the tuple
                    return tmp_tuple;
                },
                m_state);

            status = this->get_writable_edge()->await_write(this->convert_value(std::move(new_val)));

            // Signal that we have room for more data
            m_cv.notify_all();
        }

        return status;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(InputT))
        {
            WritableAcceptor<OutputT>::release_edge_connection();
        }
    }

    virtual output_t convert_value(input_tuple_t&& data) = 0;

    boost::fibers::mutex m_mutex;
    boost::fibers::condition_variable m_cv;
    size_t m_max_outstanding{64};
    size_t m_completions{0};
    std::tuple<std::queue<InputT>...> m_state;

    std::tuple<std::shared_ptr<WritableProvider<InputT>>...> m_upstream_holders;
};

template <typename...>
class Zip;

template <typename... InputT, typename OutputT>
class Zip<std::tuple<InputT...>, OutputT> : public ZipBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t        = ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;
};

// Specialization for Zip with a default output type
template <typename... InputT>
class Zip<std::tuple<InputT...>> : public ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>
{
  public:
    using base_t        = ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>;
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
class ZipTransform;

template <typename... InputT, typename OutputT>
class ZipTransform<std::tuple<InputT...>, OutputT> : public ZipBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t         = ZipBase<std::tuple<InputT...>, OutputT>;
    using input_tuple_t  = typename base_t::input_tuple_t;
    using output_t       = typename base_t::output_t;
    using transform_fn_t = std::function<output_t(input_tuple_t&&)>;

    ZipTransform(transform_fn_t transform_fn, size_t max_outstanding = 64) :
      base_t(max_outstanding),
      m_transform_fn(std::move(transform_fn))
    {}

  private:
    output_t convert_value(input_tuple_t&& data) override
    {
        return m_transform_fn(std::move(data));
    }

    transform_fn_t m_transform_fn;
};

// template <typename... TypesT>
// class Zip : public WritableAcceptor<std::tuple<TypesT...>>
// {
//     template <std::size_t... Is>
//     static auto build_ingress(Zip* self, std::index_sequence<Is...> /*unused*/)
//     {
//         return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
//     }

//   public:
//     Zip(size_t max_outstanding = 64) :
//       m_upstream_holders(build_ingress(const_cast<Zip*>(this), std::index_sequence_for<TypesT...>{})),
//       m_max_outstanding(max_outstanding)
//     {}

//     virtual ~Zip() = default;

//     template <size_t N>
//     std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, TypesT...>>> get_sink() const
//     {
//         return std::get<N>(m_upstream_holders);
//     }

//   protected:
//     template <size_t N>
//     class Upstream : public WritableProvider<NthTypeOf<N, TypesT...>>
//     {
//         using upstream_t = NthTypeOf<N, TypesT...>;

//       public:
//         Upstream(Zip& parent)
//         {
//             this->init_owned_edge(std::make_shared<InnerEdge>(parent));
//         }

//       private:
//         class InnerEdge : public edge::IEdgeWritable<NthTypeOf<N, TypesT...>>
//         {
//           public:
//             InnerEdge(Zip& parent) : m_parent(parent) {}
//             ~InnerEdge()
//             {
//                 m_parent.edge_complete();
//             }

//             virtual channel::Status await_write(upstream_t&& data)
//             {
//                 return m_parent.set_upstream_value<N>(std::move(data));
//             }

//           private:
//             Zip& m_parent;
//         };
//     };

//   private:
//     template <size_t N>
//     channel::Status set_upstream_value(NthTypeOf<N, TypesT...> value)
//     {
//         std::unique_lock<decltype(m_mutex)> lock(m_mutex);

//         // Update the current value
//         auto& nth_vector = std::get<N>(m_state);

//         // Ensure we have room, otherwise block
//         m_cv.wait(lock, [this, &nth_vector] {
//             return nth_vector.size() < m_max_outstanding;
//         });

//         // Push the new value
//         nth_vector.push(std::move(value));

//         channel::Status status = channel::Status::success;

//         bool all_have_values = std::apply(
//             [](auto const&... vec) {
//                 return (!vec.empty() && ...);
//             },
//             m_state);

//         // Check if we should push the new value
//         if (all_have_values)
//         {
//             // Pop the front of each vector and push the tuple
//             std::tuple<TypesT...> new_val = std::apply(
//                 [](auto&... vec) {
//                     // Move the front of each vector into a tuple
//                     auto tmp_tuple = std::make_tuple(std::move(vec.front())...);

//                     // Pop the front of each vector
//                     (vec.pop(), ...);

//                     // Return the tuple
//                     return tmp_tuple;
//                 },
//                 m_state);

//             status = this->get_writable_edge()->await_write(std::move(new_val));

//             // Signal that we have room for more data
//             m_cv.notify_all();
//         }

//         return status;
//     }

//     void edge_complete()
//     {
//         std::unique_lock<decltype(m_mutex)> lock(m_mutex);

//         m_completions++;

//         if (m_completions == sizeof...(TypesT))
//         {
//             WritableAcceptor<std::tuple<TypesT...>>::release_edge_connection();
//         }
//     }

//     boost::fibers::mutex m_mutex;
//     boost::fibers::condition_variable m_cv;
//     size_t m_max_outstanding{64};
//     size_t m_completions{0};
//     std::tuple<std::queue<TypesT>...> m_state;

//     std::tuple<std::shared_ptr<WritableProvider<TypesT>>...> m_upstream_holders;
// };

template <typename KeyT, typename InputT, typename OutputT = std::vector<InputT>>
class DynamicZipComponent : public MultiWritableProvider<KeyT, InputT>, public WritableAcceptor<OutputT>
{
  public:
    using input_t  = InputT;
    using output_t = OutputT;

    DynamicZipComponent(size_t max_outstanding = 64) : m_max_outstanding(max_outstanding) {}

    std::shared_ptr<edge::IWritableProvider<input_t>> get_sink(const KeyT& key) const
    {
        // Simply return an object that will set the message to upstream and go away
        return std::make_shared<Upstream>(*const_cast<DynamicZipComponent<KeyT, input_t, output_t>*>(this), key);
    }

    bool has_sink(const KeyT& key) const
    {
        return MultiWritableProvider<KeyT, input_t>::has_writable_edge(key);
    }

    void drop_edge(const KeyT& key)
    {
        MultiWritableProvider<KeyT, input_t>::release_writable_edge(key);
    }

  protected:
    class Upstream : public edge::IWritableProvider<input_t>
    {
      public:
        Upstream(DynamicZipComponent& parent, KeyT key) : m_parent(parent), m_key(key) {}

        std::shared_ptr<edge::WritableEdgeHandle> get_writable_edge_handle() const override
        {
            m_parent.m_state[m_key] = std::queue<input_t>{};

            m_parent.MultiWritableProvider<KeyT, input_t>::init_owned_edge(
                m_key,
                std::make_shared<InnerEdge>(m_parent, m_key));

            return m_parent.get_writable_edge_handle(m_key);
        }

      private:
        class InnerEdge : public edge::IEdgeWritable<input_t>
        {
          public:
            InnerEdge(DynamicZipComponent& parent, KeyT key) : m_parent(parent), m_key(key) {}
            ~InnerEdge()
            {
                m_parent.edge_complete(m_key);
            }

            channel::Status await_write(input_t&& data) override
            {
                return m_parent.set_upstream_value(m_key, std::move(data));
            }

          private:
            DynamicZipComponent& m_parent;
            KeyT m_key;
        };

        DynamicZipComponent& m_parent;
        KeyT m_key;
    };

    channel::Status set_upstream_value(KeyT key, input_t value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the current value
        auto& nth_vector = m_state.at(key);

        // Ensure we have room, otherwise block
        m_cv.wait(lock, [this, &nth_vector] {
            return nth_vector.size() < m_max_outstanding;
        });

        // Push the new value
        nth_vector.push(std::move(value));

        channel::Status status = channel::Status::success;

        bool all_have_values = true;

        // Loop over the vectors and check if they all have values
        for (auto const& [_, vec] : m_state)
        {
            if (vec.empty())
            {
                all_have_values = false;
                break;
            }
        }

        // Check if we should push the new value
        if (all_have_values)
        {
            // Pop the front of each vector and push the tuple
            std::vector<input_t> new_val;

            for (auto& [_, vec] : m_state)
            {
                new_val.push_back(std::move(vec.front()));
                vec.pop();
            }

            status = this->get_writable_edge()->await_write(this->convert_value(std::move(new_val)));

            // Signal that we have room for more data
            m_cv.notify_all();
        }

        return status;
    }

    void edge_complete(KeyT key)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Erase the key from the map. It must be there otherwise this is invalid
        CHECK_EQ(m_state.erase(key), 1) << "Inconsistent state. Key not found in map";

        // If the map is empty, release the downstream connections
        if (m_state.empty())
        {
            WritableAcceptor<output_t>::release_edge_connection();
        }
    }

    virtual output_t convert_value(std::vector<input_t>&& data)
    {
        return std::move(data);
    }

    boost::fibers::mutex m_mutex;
    boost::fibers::condition_variable m_cv;
    size_t m_max_outstanding{64};
    size_t m_completions{0};
    std::map<KeyT, std::queue<input_t>> m_state;
};

}  // namespace mrc::node
