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

#pragma once

#include "pysrf/forward.hpp"  // IWYU pragma: keep

#include <nlohmann/json_fwd.hpp>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>

#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>

// IWYU pragma: no_include <listobject.h>
// IWYU pragma: no_include <nlohmann/detail/iterators/iteration_proxy.hpp>
// IWYU pragma: no_include <nlohmann/detail/iterators/iter_impl.hpp>
// IWYU pragma: no_include <object.h>
// IWYU pragma: no_include <pybind11/detail/type_caster_base.h>
// IWYU pragma: no_include <pystate.h>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

pybind11::object cast_from_json(const nlohmann::json& source);
nlohmann::json cast_from_pyobject(const pybind11::object& source);

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
 * @brief Wraps a `pybind11::gil_scoped_acquire` with additional functionality to release the GIL before this object
 * leaves the scope. Useful to avoid unnecessary nested `gil_scoped_acquire` then `gil_scoped_release` which need to
 * grab the GIL twice
 *
 */
class AcquireGIL
{
  public:
    //   Create the object in place
    AcquireGIL() : m_gil(std::in_place) {}

    inline void inc_ref()
    {
        if (m_gil.has_value())
        {
            m_gil->inc_ref();
        }
    }

    inline void dec_ref()
    {
        if (m_gil.has_value())
        {
            m_gil->dec_ref();
        }
    }

    inline void disarm()
    {
        if (m_gil.has_value())
        {
            m_gil->disarm();
        }
    }

    /**
     * @brief Releases the GIL early. The GIL will only be released once.
     *
     */
    inline void release()
    {
        // Just delete the GIL object early
        m_gil.reset();
    }

  private:
    //   Use an optional here to allow releasing the GIL early
    std::optional<pybind11::gil_scoped_acquire> m_gil;
};

// Allows you to work with a pybind11::object that will correctly grab the GIL before destruction
class PYBIND11_EXPORT PyObjectWrapper : public pybind11::detail::object_api<PyObjectWrapper>
{
  public:
    PyObjectWrapper() = default;
    PyObjectWrapper(pybind11::object&& to_wrap);

    // Disallow copying since that would require the GIL. If it needs to be copied, use PyObjectHolder
    PyObjectWrapper(const PyObjectWrapper&) = delete;

    ~PyObjectWrapper();

    PyObjectWrapper& operator=(const PyObjectWrapper& other) = delete;

    PyObjectWrapper& operator=(PyObjectWrapper&& other) = default;

    explicit operator bool() const;

    // Returns const ref. Does not require GIL. Use at your own risk!!! Should only be used for testing
    const pybind11::handle& view_obj() const&;

    // Makes a copy of the underlying object. Requires the GIL
    pybind11::object copy_obj() const&;

    // Moves the underlying object. Does not require the GIL
    pybind11::object&& move_obj() &&;

    // Returns const ref. Used by object_api. Should not be used directly. Requires the GIL
    operator const pybind11::handle&() const&;

    // Main method to move values out of the wrapper
    operator pybind11::object&&() &&;

    // Necessary to implement the object_api interface
    PyObject* ptr() const;

  private:
    pybind11::object m_obj;
};

// Allows you to move a pybind11::object around without needing the GIL. Uses a shared_ptr under the hood to reference
// count
class PYBIND11_EXPORT PyObjectHolder : public pybind11::detail::object_api<PyObjectHolder>
{
  public:
    PyObjectHolder();
    PyObjectHolder(pybind11::object&& to_wrap);

    PyObjectHolder(const PyObjectHolder& other) = default;

    PyObjectHolder(PyObjectHolder&& other);

    PyObjectHolder& operator=(const PyObjectHolder& other);

    PyObjectHolder& operator=(PyObjectHolder&& other);

    explicit operator bool() const;

    // Returns const ref. Does not require GIL. Use at your own risk!!! Should only be used for testing
    const pybind11::handle& view_obj() const&;

    // Makes a copy of the underlying object. Requires the GIL
    pybind11::object copy_obj() const&;

    // Moves the underlying object. Does not require the GIL
    pybind11::object&& move_obj() &&;

    // Returns const ref. Used by object_api. Should not be used directly. Requires the GIL
    operator const pybind11::handle&() const&;

    // Main method to move values out of the wrapper
    operator pybind11::object&&() &&;

    operator pybind11::object&&() const&& = delete;

    // Necessary to implement the object_api interface
    PyObject* ptr() const;

  private:
    std::shared_ptr<PyObjectWrapper> m_wrapped;
};

#pragma GCC visibility pop

}  // namespace srf::pysrf
