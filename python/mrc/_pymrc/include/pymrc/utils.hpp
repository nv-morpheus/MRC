/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nlohmann/json_fwd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sys/types.h>

#include <string>
#include <typeinfo>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

pybind11::object cast_from_json(const nlohmann::json& source);

/**
 * @brief Convert a pybind11 object to a JSON object. If the object cannot be serialized, a pybind11::type_error
 * exception be thrown.
 * @param source : pybind11 object
 * @return nlohmann::json.
 */
nlohmann::json cast_from_pyobject(const pybind11::object& source);

/**
 * @brief Convert a pybind11 object to a JSON object. If the object cannot be serialized, the unserializable_handler_fn
 * will be invoked to handle the object.
 * @param source : pybind11 object
 * @param unserializable_handler_fn : unserializable_handler_fn_t
 * @return nlohmann::json.
 */
nlohmann::json cast_from_pyobject(const pybind11::object& source,
                                  unserializable_handler_fn_t unserializable_handler_fn);

void import_module_object(pybind11::module_&, const std::string&, const std::string&);
void import_module_object(pybind11::module_& dest, const pybind11::module_& mod);

// Imitates `import {mod}` syntax
void import(pybind11::module_& dest, const std::string& mod);

// Imitates `from {mod} import {attr}` syntax
void from_import(pybind11::module_& dest, const std::string& mod, const std::string& attr);

// Imitates `from {mod} import {attr} as {name}` syntax
void from_import_as(pybind11::module_& dest, const std::string& from, const std::string& import, const std::string& as);

/**
 * @brief Given a pybind11 object, attempt to extract its underlying cpp std::type_info* --
 *  if the wrapped type is something that was registered via pybind, ex: py::class_<...>(...), the return value
 *  will be non-null;
 * @param obj : pybind11 object
 * @return pointer to std::type_info object, or nullptr if none exists.
 */
const std::type_info* cpptype_info_from_object(pybind11::object& obj);

/**
 * @brief Given a pybind11 object, return the Python type name essentially the same as `str(type(obj))`
 * @param obj : pybind11 object
 * @return std::string.
 */
std::string get_py_type_name(const pybind11::object& obj);

void show_deprecation_warning(const std::string& deprecation_message, ssize_t stack_level = 1);

#pragma GCC visibility pop

}  // namespace mrc::pymrc
