/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/types.hpp"  // for python_map_t & unserializable_handler_fn_t

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>  // for PYBIND11_EXPORT, pybind11::object, type_caster

#include <cstddef>  // for size_t
#include <map>      // for map
#include <string>   // for string
#include <utility>  // for move
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <pybind11/pytypes.h>

namespace mrc::pymrc {

#pragma GCC visibility push(default)

/**
 * @brief Immutable container for holding Python values as JSON objects if possible, and as pybind11::object otherwise.
 * The container can be copied and moved, but the underlying JSON object is immutable.
 **/
class PYBIND11_EXPORT JSONValues
{
  public:
    JSONValues();
    JSONValues(pybind11::object values);
    JSONValues(nlohmann::json values);

    JSONValues(const JSONValues& other) = default;
    JSONValues(JSONValues&& other)      = default;
    ~JSONValues()                       = default;

    JSONValues& operator=(const JSONValues& other) = default;
    JSONValues& operator=(JSONValues&& other)      = default;

    /**
     * @brief Sets a value in the JSON object at the specified path with the provided Python object. If `value` is
     * serializable as JSON it will be stored as JSON, otherwise it will be stored as-is.
     * @param path The path in the JSON object where the value should be set.
     * @param value The Python object to set.
     * @throws std::runtime_error If the path is invalid.
     * @return A new JSONValues object with the updated value.
     */
    JSONValues set_value(const std::string& path, const pybind11::object& value) const;

    /**
     * @brief Sets a value in the JSON object at the specified path with the provided JSON object.
     * @param path The path in the JSON object where the value should be set.
     * @param value The JSON object to set.
     * @throws std::runtime_error If the path is invalid.
     * @return A new JSONValues object with the updated value.
     */
    JSONValues set_value(const std::string& path, nlohmann::json value) const;

    /**
     * @brief Sets a value in the JSON object at the specified path with the provided JSONValues object.
     * @param path The path in the JSON object where the value should be set.
     * @param value The JSONValues object to set.
     * @throws std::runtime_error If the path is invalid.
     * @return A new JSONValues object with the updated value.
     */
    JSONValues set_value(const std::string& path, const JSONValues& value) const;

    /**
     * @brief Returns the number of unserializable Python objects.
     * @return The number of unserializable Python objects.
     */
    std::size_t num_unserializable() const;

    /**
     * @brief Checks if there are any unserializable Python objects.
     * @return True if there are unserializable Python objects, false otherwise.
     */
    bool has_unserializable() const;

    /**
     * @brief Convert to a Python object.
     * @return The Python object representation of the values.
     */
    pybind11::object to_python() const;

    /**
     * @brief Returns a constant reference to the underlying JSON object. Any unserializable Python objects, will be
     * represented in the JSON object with a string place-holder with the value `"**pymrc_placeholder"`.
     * @return A constant reference to the JSON object.
     */
    nlohmann::json::const_reference view_json() const;

    /**
     * @brief Converts the JSON object to a JSON object. If any unserializable Python objects are present, the
     * `unserializable_handler_fn` will be invoked to handle the object.
     * @param unserializable_handler_fn Optional function to handle unserializable objects.
     * @return The JSON string representation of the JSON object.
     */
    nlohmann::json to_json(unserializable_handler_fn_t unserializable_handler_fn) const;

    /**
     * @brief Converts a Python object to a JSON string. Convienence function that matches the
     * `unserializable_handler_fn_t` signature. Convienent for use with `to_json` and `get_json`.
     * @param obj The Python object to convert.
     * @param path The path in the JSON object where the value should be set.
     * @return The JSON string representation of the Python object.
     */
    static nlohmann::json stringify(const pybind11::object& obj, const std::string& path);

    /**
     * @brief Returns the object at the specified path as a Python object.
     * @param path Path to the specified object.
     * @throws std::runtime_error If the path does not exist or is not a valid path.
     * @return Python representation of the object at the specified path.
     */
    pybind11::object get_python(const std::string& path) const;

    /**
     * @brief Returns the object at the specified path. If the object is an unserializable Python object the
     * `unserializable_handler_fn` will be invoked.
     * @param path Path to the specified object.
     * @param unserializable_handler_fn Function to handle unserializable objects.
     * @throws std::runtime_error If the path does not exist or is not a valid path.
     * @return The JSON object at the specified path.
     */
    nlohmann::json get_json(const std::string& path, unserializable_handler_fn_t unserializable_handler_fn) const;

    /**
     * @brief Return a new JSONValues object with the value at the specified path.
     * @param path Path to the specified object.
     * @throws std::runtime_error If the path does not exist or is not a valid path.
     * @return The value at the specified path.
     */
    JSONValues operator[](const std::string& path) const;

  private:
    JSONValues(nlohmann::json&& values, python_map_t&& py_objects);
    nlohmann::json unserializable_handler(const pybind11::object& obj, const std::string& path);

    nlohmann::json m_serialized_values;
    python_map_t m_py_objects;
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc

/****** Pybind11 caster ******************/

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<mrc::pymrc::JSONValues>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type JSONValues
     */
    PYBIND11_TYPE_CASTER(mrc::pymrc::JSONValues, _("object"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into JSONValues
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = mrc::pymrc::JSONValues();
        }
        else
        {
            value = std::move(mrc::pymrc::JSONValues(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert a JSONValues instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(mrc::pymrc::JSONValues src, return_value_policy policy, handle parent)
    {
        return src.to_python().release();
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
