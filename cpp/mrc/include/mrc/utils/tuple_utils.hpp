#pragma once

#include <tuple>

namespace mrc::utils {

template <typename TupleT, std::size_t... Is>
auto tuple_surely(TupleT&& tuple, std::index_sequence<Is...> /*unused*/)
{
    return std::tuple<typename std::tuple_element_t<Is, std::decay_t<TupleT>>::value_type...>(
        (std::get<Is>(tuple).value())...);
}

/**
 * @brief Converts a std::tuple<std::optional<T1>, std::optional<T2>, ...> to std::tuple<T1, T1, ...>
 *
 * @tparam TupleT The type of tuple
 * @param tuple
 * @return auto A new Tuple with `std::optional` types removed
 */
template <typename TupleT>
auto tuple_surely(TupleT&& tuple)
{
    return tuple_surely(std::forward<TupleT>(tuple),
                        std::make_index_sequence<std::tuple_size<std::decay_t<TupleT>>::value>());
}

template <typename TupleT, typename FuncT, std::size_t... Is>
void tuple_for_each(TupleT&& tuple, FuncT&& f, std::index_sequence<Is...> /*unused*/)
{
    (f(std::get<Is>(std::forward<TupleT>(tuple)), Is), ...);
}

/**
 * @brief Executes a function for each element of a tuple.
 *
 * @tparam TupleT The type of the tuple
 * @tparam FuncT The type of the lambda
 * @param tuple Tuple to run the function on
 * @param f A function which accepts an element of the tuple as the first arg and the index for the second arg.
 * Recommended to use `auto` or a templated lambda as the first argument
 */
template <typename TupleT, typename FuncT>
void tuple_for_each(TupleT&& tuple, FuncT&& f)
{
    tuple_for_each(std::forward<TupleT>(tuple),
                   std::forward<FuncT>(f),
                   std::make_index_sequence<std::tuple_size<std::decay_t<TupleT>>::value>());
}
}  // namespace mrc::utils
