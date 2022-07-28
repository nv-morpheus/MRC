/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pysrf/utils.hpp"

#include "srf/exceptions/runtime_error.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

// IWYU pragma: no_include <listobject.h>
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <nlohmann/detail/iterators/iteration_proxy.hpp>
// IWYU pragma: no_include <nlohmann/detail/iterators/iter_impl.hpp>
// IWYU pragma: no_include "object.h"
// IWYU pragma: no_include <pybind11/detail/type_caster_base.h>
// IWYU pragma: no_include "pystate.h"

namespace srf::pysrf {

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
        return py::str(source.get<std::string>());
    }

    return py::none();
    // throw std::runtime_error("Unsupported conversion type.");
}

json cast_from_pyobject(const py::object& source)
{
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

    // else unsupported return null
    return json();
}

PyObjectWrapper::PyObjectWrapper(pybind11::object&& to_wrap) : m_obj(std::move(to_wrap)) {}

PyObjectWrapper::~PyObjectWrapper()
{
    // If we are being destroyed with a wrapped object, grab the GIL before destroying
    if (m_obj)
    {
        // Two paths here make it easy to put breakpoints in for debugging
        if (PyGILState_Check() == 0)
        {
            pybind11::gil_scoped_acquire gil;

            pybind11::object tmp = std::move(m_obj);
        }
        else
        {
            pybind11::object tmp = std::move(m_obj);
        }

        assert(!m_obj);
    }
}

PyObjectWrapper::operator bool() const
{
    return (bool)m_obj;
}

const pybind11::handle& PyObjectWrapper::view_obj() const&
{
    // Allow for peaking into the object
    return m_obj;
}

pybind11::object PyObjectWrapper::copy_obj() const&
{
    if (PyGILState_Check() == 0)
    {
        throw srf::exceptions::SrfRuntimeError("Must have the GIL copying to py::object");
    }

    // Allow for peaking into the object
    return py::object(m_obj);
}

pybind11::object&& PyObjectWrapper::move_obj() &&
{
    if (!m_obj)
    {
        throw srf::exceptions::SrfRuntimeError(
            "Cannot convert empty wrapper to py::object. Did you accidentally move out the object?");
    }

    return std::move(m_obj);
}

PyObjectWrapper::operator const pybind11::handle&() const&
{
    return m_obj;
}

PyObjectWrapper::operator pybind11::object&&() &&
{
    return std::move(m_obj);
}

PyObject* PyObjectWrapper::ptr() const
{
    return m_obj.ptr();
}

PyObjectHolder::PyObjectHolder() : m_wrapped(std::make_shared<PyObjectWrapper>()) {}

PyObjectHolder::PyObjectHolder(pybind11::object&& to_wrap) :
  m_wrapped(std::make_shared<PyObjectWrapper>(std::move(to_wrap)))
{}

PyObjectHolder::PyObjectHolder(PyObjectHolder&& other) : m_wrapped(std::move(other.m_wrapped)) {}

PyObjectHolder& PyObjectHolder::operator=(const PyObjectHolder& other)
{
    if (this == &other)
    {
        return *this;
    }

    m_wrapped = other.m_wrapped;

    return *this;
}

PyObjectHolder& PyObjectHolder::operator=(PyObjectHolder&& other)
{
    if (this == &other)
    {
        return *this;
    }

    m_wrapped.reset();
    std::swap(m_wrapped, other.m_wrapped);

    return *this;
}

PyObjectHolder::operator bool() const
{
    return (bool)*m_wrapped;
}

const pybind11::handle& PyObjectHolder::view_obj() const&
{
    // Allow for peaking into the object
    return m_wrapped->view_obj();
}

pybind11::object PyObjectHolder::copy_obj() const&
{
    // Allow for peaking into the object
    return m_wrapped->copy_obj();
}

pybind11::object&& PyObjectHolder::move_obj() &&
{
    return std::move(*m_wrapped).move_obj();
}

PyObjectHolder::operator const pybind11::handle&() const&
{
    // TODO(MDD): Do we need the GIL here?
    if (PyGILState_Check() == 0)
    {
        throw srf::exceptions::SrfRuntimeError("Must have the GIL copying to py::object");
    }

    return m_wrapped->view_obj();
}

PyObjectHolder::operator pybind11::object&&() &&
{
    return std::move(*m_wrapped).move_obj();
}

PyObject* PyObjectHolder::ptr() const
{
    return m_wrapped->ptr();
}

}  // namespace srf::pysrf
