/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/utils.hpp"

#include "pymrc/utilities/object_cache.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>
#include <warnings.h>

#include <regex>
#include <string>
#include <utility>

namespace mrc::pymrc {

namespace py = pybind11;

using nlohmann::json;

void import_module_object(py::module_& dest, const std::string& source, const std::string& member)
{
    dest.attr(member.c_str()) = py::module_::import(source.c_str()).attr(member.c_str());
}

void import_module_object(py::module_& dest, const py::module_& mod)
{
    // Get the module name and save in __dict__
    auto mod_name               = mod.attr("__name__")().cast<std::string>();
    dest.attr(mod_name.c_str()) = mod;
}

void import(pybind11::module_& dest, const std::string& mod)
{
    dest.attr(mod.c_str()) = py::module_::import(mod.c_str());
}

void from_import(pybind11::module_& dest, const std::string& mod, const std::string& attr)
{
    from_import_as(dest, mod, attr, attr);
}

void from_import_as(py::module_& dest, const std::string& from, const std::string& import, const std::string& as)
{
    dest.attr(as.c_str()) = py::module_::import(from.c_str()).attr(import.c_str());
}

const std::type_info* cpptype_info_from_object(py::object& obj)
{
    py::detail::type_info* tinfo = py::detail::get_type_info((PyTypeObject*)obj.ptr());
    if (tinfo != nullptr)
    {
        return tinfo->cpptype;
    }

    return nullptr;
}

py::object cast_from_json(const json& source)
{
    if (source.is_null())
    {
        return py::none();
    }

    if (source.is_array())
    {
        py::list list_;
        for (const auto& element : source)
        {
            list_.append(cast_from_json(element));
        }
        return std::move(list_);
    }

    if (source.is_boolean())
    {
        return py::bool_(source.get<bool>());
    }

    if (source.is_number_float())
    {
        return py::float_(source.get<double>());
    }

    if (source.is_number_integer())
    {
        return py::int_(source.get<json::number_integer_t>());
    }

    if (source.is_number_unsigned())
    {
        return py::int_(source.get<json::number_unsigned_t>());  // std::size_t ?
    }

    if (source.is_object())
    {
        py::dict dict;
        for (const auto& it : source.items())
        {
            dict[py::str(it.key())] = cast_from_json(it.value());
        }

        return std::move(dict);
    }

    if (source.is_string())
    {
        std::string str_val = source.get<std::string>();
        std::regex uuid_regex("cache_object:([0-9a-fA-F-]{36})");
        std::smatch uuid_match;

        if (std::regex_search(str_val, uuid_match, uuid_regex))
        {
            std::string uuid = uuid_match[1];

            auto& cache      = PythonObjectCache::get_handle();
            if (cache.contains(uuid))
            {
                return cache.get(uuid);
            } else {
                throw std::runtime_error("Cached object id not found in cache: " + uuid);
            }
        }

        return py::str(str_val);
    }

    return py::none();
    // throw std::runtime_error("Unsupported conversion type.");
}

json cast_from_pyobject(const py::object& source)
{
    // Dont return via initializer list with JSON. It performs type deduction and gives different results
    // NOLINTBEGIN(modernize-return-braced-init-list)
    if (source.is_none())
    {
        return json();
    }

    if (py::isinstance<py::dict>(source))
    {
        const auto py_dict = source.cast<py::dict>();
        auto json_obj      = json::object();
        for (const auto& p : py_dict)
        {
            json_obj[py::cast<std::string>(p.first)] = cast_from_pyobject(p.second.cast<py::object>());
        }

        return json_obj;
    }

    if (py::isinstance<py::list>(source) || py::isinstance<py::tuple>(source))
    {
        const auto py_list = source.cast<py::list>();
        auto json_arr      = json::array();
        for (const auto& p : py_list)
        {
            json_arr.push_back(cast_from_pyobject(p.cast<py::object>()));
        }

        return json_arr;
    }

    if (py::isinstance<py::bool_>(source))
    {
        return json(py::cast<bool>(source));
    }

    if (py::isinstance<py::int_>(source))
    {
        return json(py::cast<long>(source));
    }

    if (py::isinstance<py::float_>(source))
    {
        return json(py::cast<double>(source));
    }

    if (py::isinstance<py::str>(source))
    {
        return json(py::cast<std::string>(source));
    }

    /* We don't know how to serialize the Object, throw it into cache and return a reference ID*/
    // Use Python's uuid module to generate a UUID
    py::object uuid_module = py::module_::import("uuid");
    py::object uuid_obj    = uuid_module.attr("uuid4")();
    std::string uuid_str   = py::str(uuid_obj);

    // Remove constness and cache the object
    py::object non_const_source = const_cast<py::object&>(source);
    PythonObjectCache::get_handle().cache_object(uuid_str, non_const_source);

    // Return the UUID string
    return json(std::string("cache_object:") + uuid_str);
    // NOLINTEND(modernize-return-braced-init-list)
}

void show_deprecation_warning(const std::string& deprecation_message, ssize_t stack_level)
{
    PyErr_WarnEx(PyExc_DeprecationWarning, deprecation_message.c_str(), stack_level);
}
}  // namespace mrc::pymrc
