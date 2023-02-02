//
// Created by drobison on 1/20/23.
//

#pragma once

#include "mrc/segment/object.hpp"
#include "mrc/type_traits.hpp"

#include <concepts>

namespace mrc {

template <typename T>
struct is_mrc_object_type : public std::false_type
{};

template <typename T>
struct is_mrc_object_type<mrc::segment::Object<T>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_mrc_object_v = is_mrc_object_type<T>::value;  // NOLINT

template <typename T>
struct is_mrc_object_shared_pointer : public std::false_type
{};

template <typename T>
struct is_mrc_object_shared_pointer<std::shared_ptr<mrc::segment::Object<T>>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_mrc_object_shared_ptr_v = is_mrc_object_shared_pointer<T>::value;  // NOLINT

struct mrc_object_null_type
{
    using source_type_t = std::nullptr_t;
    using sink_type_t   = std::nullptr_t;
};

template <typename T>
struct mrc_object_sptr_type
{
    using type_t = mrc_object_null_type;
};

template <typename T>
struct mrc_object_sptr_type<std::shared_ptr<mrc::segment::Object<T>>>
{
    using type_t = T;
};

template <typename T>
using mrc_object_sptr_type_t = typename mrc_object_sptr_type<T>::type_t;

template <typename TypeT>
concept MRCObject = is_mrc_object_v<TypeT>;

template <typename TypeT>
concept MRCObjectSharedPtr = is_mrc_object_shared_ptr_v<TypeT>;

template <typename TypeT>
concept MRCObjProp = std::is_same_v<std::decay_t<TypeT>, mrc::segment::ObjectProperties>;

template <typename TypeT>
concept MRCObjPropSharedPtr = std::is_same_v<std::decay_t<TypeT>, std::shared_ptr<mrc::segment::ObjectProperties>>;

template <typename TypeT>
concept MRCObjectProxy = MRCObject<TypeT> || MRCObjectSharedPtr<TypeT> || MRCObjProp<TypeT> ||
                         MRCObjPropSharedPtr<TypeT> || std::is_convertible_v<TypeT, std::string>;

}  // namespace mrc