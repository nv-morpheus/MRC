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

#include "pymrc/utilities/function_wrappers.hpp"

#include "pymrc/utilities/object_wrappers.hpp"

namespace mrc::pymrc {

namespace py = pybind11;

PyFuncWrapper::PyFuncWrapper(pybind11::function&& fn) : m_fn(std::move(fn))
{
    // Save the name of the function to help debugging
    if (m_fn)
    {
        m_repr = pybind11::str(m_fn);
    }
}

PyFuncWrapper::~PyFuncWrapper()
{
    pybind11::gil_scoped_acquire acq;
    pybind11::function kill_f(std::move(m_fn));
}

PyFuncWrapper::PyFuncWrapper(const PyFuncWrapper& other)
{
    operator=(other);
}

PyFuncWrapper& PyFuncWrapper::operator=(const PyFuncWrapper& other)
{
    pybind11::gil_scoped_acquire acq;
    m_fn   = other.m_fn;
    m_repr = other.m_repr;
    return *this;
}

const pybind11::function& PyFuncWrapper::py_function_obj() const
{
    return m_fn;
}

pybind11::function& PyFuncWrapper::py_function_obj()
{
    pybind11::gil_scoped_acquire acq;

    return m_fn;
}

const std::string& PyFuncWrapper::repr() const
{
    return m_repr;
}

OnNextFunction::cpp_fn_t OnNextFunction::build_cpp_function(pybind11::function&& py_fn) const
{
    if (!py_fn)
    {
        return [](PyObjectHolder x) {
            pybind11::gil_scoped_acquire gil;

            // Kill the object with the GIL held
            pybind11::object kill(std::move(x));
        };
    }

    pybind11::module_ inspect = pybind11::module_::import("inspect");

    auto signature_fn = inspect.attr("signature");

    // Get the number of args for the supplied function
    auto number_of_args = pybind11::len(signature_fn(py_fn).attr("parameters"));

    if (number_of_args == 0)
    {
        throw std::runtime_error(MRC_CONCAT_STR("Python on_next function '" << std::string(pybind11::str(py_fn))
                                                                            << "', must accept at least one argument"));
        return nullptr;
    }

    // No need to unpack
    if (number_of_args == 1)
    {
        // Return the base implementation
        return base_t::build_cpp_function(std::move(py_fn));
    }

    // Wrap the original py::function for safe cleanup
    return [holder = PyFuncWrapper(std::move(py_fn))](PyObjectHolder x) -> void {
        pybind11::gil_scoped_acquire gil;

        // Move it into a temporary object
        pybind11::object obj = std::move(x);

        // Unpack the arguments
        holder.operator()<void, pybind11::detail::args_proxy>(*obj);
    };
}

OnErrorFunction::cpp_fn_t OnErrorFunction::build_cpp_function(pybind11::function&& py_fn) const
{
    if (!py_fn)
    {
        return [](std::exception_ptr x) {
            // Do nothing. Object will go out of scope and the holder will decrement the reference
        };
    }

    return [holder = PyFuncWrapper(std::move(py_fn))](std::exception_ptr x) {
        pybind11::gil_scoped_acquire gil;

        // First, translate the exception setting the python exception value
        pybind11::detail::translate_exception(x);

        // Creating py::error_already_set will clear the exception and retrieve the value
        pybind11::error_already_set active_ex;

        // Now actually pass the exception to the callback
        holder.operator()<void, pybind11::object>(active_ex.value());
    };
}

OnDataFunction::cpp_fn_t OnDataFunction::build_cpp_function(pybind11::function&& py_fn) const
{
    if (!py_fn)
    {
        throw std::runtime_error(MRC_CONCAT_STR("Python on_data function argument '"
                                                << std::string(pybind11::str(py_fn))
                                                << "', cannot be None since it returns a value"));
    }

    pybind11::module_ inspect = pybind11::module_::import("inspect");

    auto signature_fn = inspect.attr("signature");

    // Get the number of args for the supplied function
    auto number_of_args = pybind11::len(signature_fn(py_fn).attr("parameters"));

    if (number_of_args == 0)
    {
        throw std::runtime_error(MRC_CONCAT_STR("Python on_next function '" << std::string(pybind11::str(py_fn))
                                                                            << "', must accept at least one argument"));
        return nullptr;
    }

    // No need to unpack
    if (number_of_args == 1)
    {
        // Return the base implementation
        return base_t::build_cpp_function(std::move(py_fn));
    }

    // Wrap the original py::function for safe cleanup
    return [holder = PyFuncWrapper(std::move(py_fn))](PyObjectHolder x) -> PyObjectHolder {
        pybind11::gil_scoped_acquire gil;

        // Move it into a temporary object
        pybind11::object obj = std::move(x);

        // Unpack the arguments
        return holder.operator()<pybind11::object, pybind11::detail::args_proxy>(*obj);
    };
}

}  // namespace mrc::pymrc
