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

#include "pymrc/types.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/pytypes.h>  // for PYBIND11_EXPORT & pybind11::object

#include <cstddef>  // for size_t
#include <string>
// IWYU wants us to use the pybind11.h for the PYBIND11_EXPORT macro, but we already have it in pytypes.h
// IWYU pragma: no_include <pybind11/pybind11.h>

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

    // TODO: Docstrings
    JSONValues set_value(const std::string& path, const pybind11::object& value) const;
    JSONValues set_value(const std::string& path, nlohmann::json value) const;
    JSONValues set_value(const std::string& path, const JSONValues& value) const;

    std::size_t num_unserializable() const;
    bool has_unserializable() const;

    pybind11::object to_python() const;

    nlohmann::json::const_reference view_json() const;

    nlohmann::json to_json(unserializable_handler_fn_t unserializable_handler_fn = nullptr) const;

    static nlohmann::json stringify(const pybind11::object& obj, const std::string& path);

    pybind11::object get_python(const std::string& path) const;
    nlohmann::json get_json(const std::string& path,
                            unserializable_handler_fn_t unserializable_handler_fn = nullptr) const;

    JSONValues operator[](const std::string& path) const;

  private:
    JSONValues(nlohmann::json&& values, python_map_t&& py_objects);
    nlohmann::json unserializable_handler(const pybind11::object& obj, const std::string& path);

    nlohmann::json m_serialized_values;
    python_map_t m_py_objects;
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
