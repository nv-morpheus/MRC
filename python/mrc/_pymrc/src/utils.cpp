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

#include "pymrc/utils.hpp"

#include "mrc/exceptions/runtime_error.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cassert>
#include <memory>
#include <string>
#include <utility>

namespace mrc::pymrc {

namespace py = pybind11;
using nlohmann::json;

// Taken straight from pybind11 function cast
struct PyFuncHandle
{
    py::function f;
#if !(defined(_MSC_VER) && _MSC_VER == 1916 && defined(PYBIND11_CPP17))
    // This triggers a syntax error under very special conditions (very weird indeed).
    explicit
#endif
        PyFuncHandle(py::function&& fn) noexcept :
      f(std::move(fn))
    {
        this->m_repr = py::str(this->f);
    }
    PyFuncHandle(const PyFuncHandle& other)
    {
        operator=(other);
    }
    PyFuncHandle& operator=(const PyFuncHandle& other)
    {
        py::gil_scoped_acquire acq;
        f      = other.f;
        m_repr = other.m_repr;
        return *this;
    }
    ~PyFuncHandle()
    {
        py::gil_scoped_acquire acq;
        py::function kill_f(std::move(f));
    }

    std::string m_repr;
};

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

AcquireGIL::AcquireGIL() : m_gil(std::make_unique<py::gil_scoped_acquire>()) {}

AcquireGIL::~AcquireGIL() = default;

inline void AcquireGIL::inc_ref()
{
    if (m_gil)
    {
        m_gil->inc_ref();
    }
}

inline void AcquireGIL::dec_ref()
{
    if (m_gil)
    {
        m_gil->dec_ref();
    }
}

void AcquireGIL::disarm()
{
    if (m_gil)
    {
        m_gil->disarm();
    }
}

void AcquireGIL::release()
{
    // Just delete the GIL object early
    m_gil.reset();
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
        throw mrc::exceptions::MrcRuntimeError("Must have the GIL copying to py::object");
    }

    // Allow for peaking into the object
    return py::object(m_obj);
}

pybind11::object&& PyObjectWrapper::move_obj() &&
{
    if (!m_obj)
    {
        throw mrc::exceptions::MrcRuntimeError(
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
        throw mrc::exceptions::MrcRuntimeError("Must have the GIL copying to py::object");
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

std::function<void(PyObjectHolder)> wrap_py_on_next(py::function py_fn)
{
    if (!py_fn)
    {
        return nullptr;
        // return [](py::object x) {
        //     // Grab the GIL and decrement the ref count
        //     py::gil_scoped_acquire gil;

        //     auto tmp = std::move(x);

        //     return py::none();
        // };
    }

    py::module_ inspect = py::module_::import("inspect");

    auto signature_fn = inspect.attr("signature");

    // Get the number of args for the supplied function
    auto number_of_args = py::len(signature_fn(py_fn).attr("parameters"));

    if (number_of_args == 0)
    {
        // raise error
        LOG(ERROR) << "Python on_next function must accept at least one argument";
        return nullptr;
    }

    // No need to unpack
    if (number_of_args == 1)
    {
        // Wrap the original py::function for safe cleanup
        return [wrapper = PyFuncHandle(std::move(py_fn))](PyObjectHolder x) {
            py::gil_scoped_acquire gil;

            // Unpack the arguments
            wrapper.f(std::move(x));
        };
    }

    // Wrap the original py::function for safe cleanup
    return [wrapper = PyFuncHandle(std::move(py_fn))](PyObjectHolder x) {
        py::gil_scoped_acquire gil;

        // Move it into a temporary object
        py::object obj = std::move(x);

        // Unpack the arguments
        wrapper.f(*obj);
    };
}

std::function<void(std::exception_ptr)> wrap_py_on_error(py::function py_fn)
{
    if (!py_fn)
    {
        return nullptr;
        // // Return an empty function
        // return [](std::exception_ptr _) {
        //     // TODO: Add unhandled
        // };
    }

    return [wrapper = PyFuncHandle(std::move(py_fn))](std::exception_ptr x) {
        pybind11::gil_scoped_acquire gil;

        // First, translate the exception setting the python exception value
        py::detail::translate_exception(x);

        // Creating py::error_already_set will clear the exception and retrieve the value
        py::error_already_set active_ex;

        // Now actually pass the exception to the callback
        wrapper.f(active_ex.value());
    };
}

std::function<void()> wrap_py_on_completed(py::function py_fn)
{
    if (!py_fn)
    {
        // Return an empty function
        return nullptr;
    }

    // Otherwise cast it
    return [wrapper = PyFuncHandle(std::move(py_fn))]() {
        py::gil_scoped_acquire gil;

        // Unpack the arguments
        wrapper.f();
    };
}

}  // namespace mrc::pymrc
