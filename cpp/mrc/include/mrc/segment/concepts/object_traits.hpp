//
// Created by drobison on 1/20/23.
//

#pragma once

#include "mrc/segment/object.hpp"
#include "mrc/type_traits.hpp"

#include <concepts>

template <typename T>
struct is_object : public std::false_type
{};

template <typename T>
struct is_object<mrc::segment::Object<T>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_object_v = is_object<T>::value;  // NOLINT

template <typename T>
struct is_object_shared_ptr : public std::false_type
{};

template <typename T>
struct is_object_shared_ptr<std::shared_ptr<mrc::segment::Object<T>>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_object_shared_ptr_v = is_object_shared_ptr<T>::value;  // NOLINT

struct ObjectNullType
{
    using source_type_t = std::nullptr_t;
    using sink_type_t   = std::nullptr_t;
};

template <typename T>
struct ObjectSharedPtrType
{
    using type_t = ObjectNullType;
};

template <typename T>
struct ObjectSharedPtrType<std::shared_ptr<mrc::segment::Object<T>>>
{
    using type_t = T;
};

template <typename TypeT>
concept MRCObject = is_object_v<TypeT>;

template <typename TypeT>
concept MRCObjectSharedPtr = is_object_shared_ptr_v<TypeT>;

template <typename TypeT>
concept MRCObjectProperties = std::is_same_v<TypeT, mrc::segment::ObjectProperties>;

template <typename TypeT>
concept MRCObjectPropertiesSharedPtr = std::is_same_v<TypeT, std::shared_ptr<mrc::segment::ObjectProperties>>;

template <typename TypeT>
concept MRCObjectPropRepr = MRCObject<TypeT> || MRCObjectSharedPtr<TypeT> || MRCObjectProperties<TypeT> ||
                              MRCObjectPropertiesSharedPtr<TypeT> || std::is_convertible_v<TypeT, std::string>;
