#pragma once

#include "mrc/channel/status.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/mutex.hpp>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace mrc::node {

// template <typename... TypesT>
// class ParameterPackIndexer
// {
//   public:
//     ParameterPackIndexer(TypesT... ts) : ParameterPackIndexer(std::make_index_sequence<sizeof...(TypesT)>{}, ts...)
//     {}

//     std::tuple<std::tuple<TypesT, std::size_t>...> tup;

//   private:
//     template <std::size_t... Is>
//     ParameterPackIndexer(std::index_sequence<Is...> const& /*unused*/, TypesT... ts) : tup{std::make_tuple(ts,
//     Is)...}
//     {}
// };

// template <typename TargetT, typename ListHeadT, typename... ListTailsT>
// constexpr size_t getTypeIndexInTemplateList()
// {
//     if constexpr (std::is_same<TargetT, ListHeadT>::value)
//     {
//         return 0;
//     }
//     else
//     {
//         return 1 + getTypeIndexInTemplateList<TargetT, ListTailsT...>();
//     }
// }

namespace detail {
struct Surely
{
    template <class... T>
    auto operator()(const T&... t) const -> decltype(std::make_tuple(t.value()...))
    {
        return std::make_tuple(t.value()...);
    }
};
}  // namespace detail

// template <class... T>
// inline auto surely(const std::tuple<T...>& tpl) -> decltype(rxcpp::util::apply(tpl, detail::surely()))
// {
//     return rxcpp::util::apply(tpl, detail::surely());
// }

template <class... T>
inline auto surely2(const std::tuple<T...>& tpl)
{
    return std::apply([](auto... args) { return std::make_tuple(args.value()...); });
}

// template <typename... TypesT>
// static auto surely2(const std::tuple<TypesT...>& tpl, std::index_sequence<Is...>)
// {
//     return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
// }

// template <size_t i, typename T>
// struct IndexTypePair
// {
//     static constexpr size_t index{i};
//     using Type = T;
// };

// template <typename... T>
// struct make_index_type_tuple_helper
// {
//     template <typename V>
//     struct idx;

//     template <size_t... Indices>
//     struct idx<std::index_sequence<Indices...>>
//     {
//         using tuple_type = std::tuple<IndexTypePair<Indices, T>...>;
//     };

//     using tuple_type = typename idx<std::make_index_sequence<sizeof...(T)>>::tuple_type;
// };

// template <typename... T>
// using make_index_type_tuple = typename make_index_type_tuple_helper<T...>::tuple_type;

template <typename... TypesT>
class CombineLatest : public IngressAcceptor<std::tuple<TypesT...>>
{
    template <std::size_t... Is>
    static auto build_ingress(CombineLatest* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

  public:
    CombineLatest() :
      m_upstream_holders(build_ingress(const_cast<CombineLatest*>(this), std::index_sequence_for<TypesT...>{}))
    {
        // auto a = build_ingress(const_cast<CombineLatest*>(this), std::index_sequence_for<TypesT...>{});
    }

    virtual ~CombineLatest() = default;

    template <size_t N>
    std::shared_ptr<IIngressProvider<NthTypeOf<N, TypesT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

  protected:
    template <size_t N>
    class Upstream : public IngressProvider<NthTypeOf<N, TypesT...>>
    {
        using upstream_t = NthTypeOf<N, TypesT...>;

      public:
        Upstream(CombineLatest& parent)
        {
            this->init_owned_edge(std::make_shared<InnerEdge>(parent));
        }

        // ~Upstream()
        // {
        //     m_parent.edge_complete();
        // }

        // virtual channel::Status await_write(upstream_t&& data)
        // {
        //     return m_parent.set_upstream_value<N>(std::move(data));
        // }

        // std::shared_ptr<IngressHandleObj> get_ingress_obj() const override
        // {
        //     return std::make_shared<IngressHandleObj>(const_cast<UpstreamEdge*>(this)->shared_from_this());
        // }

      private:
        class InnerEdge : public IEdgeWritable<NthTypeOf<N, TypesT...>>
        {
          public:
            InnerEdge(CombineLatest& parent) : m_parent(parent) {}
            ~InnerEdge()
            {
                m_parent.edge_complete();
            }

            virtual channel::Status await_write(upstream_t&& data)
            {
                return m_parent.set_upstream_value<N>(std::move(data));
            }

          private:
            CombineLatest& m_parent;
        };

        // CombineLatest& m_parent;
    };

  private:
    template <size_t N>
    channel::Status set_upstream_value(NthTypeOf<N, TypesT...> value)
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
        if (m_values_set == sizeof...(TypesT))
        {
            // std::tuple<TypesT...> new_val = surely2(m_state);

            // status = this->get_writable_edge()->await_write(std::move(new_val));
        }

        return status;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(TypesT))
        {
            IngressAcceptor<std::tuple<TypesT...>>::release_edge_connection();
        }
    }

    boost::fibers::mutex m_mutex;
    size_t m_values_set{0};
    size_t m_completions{0};
    std::tuple<std::optional<TypesT>...> m_state;

    std::tuple<std::shared_ptr<IngressProvider<TypesT>>...> m_upstream_holders;
};

}  // namespace mrc::node
