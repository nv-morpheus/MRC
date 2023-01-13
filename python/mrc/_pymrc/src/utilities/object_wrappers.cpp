/**
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

#include "pymrc/utilities/object_wrappers.hpp"

#include "mrc/exceptions/runtime_error.hpp"

#include <type_traits>
#include <utility>

namespace mrc::pymrc {

namespace py = pybind11;

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
    return {m_obj};
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

PyObjectHolder::operator const pybind11::handle&() const&
{
    if (PyGILState_Check() == 0)
    {
        throw mrc::exceptions::MrcRuntimeError("Must have the GIL copying to py::object");
    }

    return m_wrapped->view_obj();
}

PyObjectHolder::operator pybind11::object() &&
{
    // If we are converting to a pybind11::object, then we need to lose the m_wrapped shared_ptr. Make sure to reset to
    // a new object since m_wrapped should never be null
    auto tmp = std::exchange(m_wrapped, std::make_shared<PyObjectWrapper>());

    // Return a copy before exiting. This will increment the ref count before the m_wrapped goes away (which could
    // decrement the ref count)
    return tmp->copy_obj();
}

PyObject* PyObjectHolder::ptr() const
{
    return m_wrapped->ptr();
}

}  // namespace mrc::pymrc

namespace pybind11::detail {

bool detail::type_caster<mrc::pymrc::PyObjectHolder>::load(handle src, bool convert)
{
    value = reinterpret_borrow<object>(src);

    return true;
}

handle pybind11::detail::type_caster<mrc::pymrc::PyObjectHolder>::cast(mrc::pymrc::PyObjectHolder src,
                                                                       return_value_policy /* policy */,
                                                                       handle /* parent */)
{
    // Since the PyObjectHolder is going out of scope, this could potentially decrement the ref. Increment it here
    // before returning
    return src.view_obj().inc_ref();
}

}  // namespace pybind11::detail
