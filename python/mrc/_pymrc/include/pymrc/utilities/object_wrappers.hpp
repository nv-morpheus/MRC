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

#include "mrc/utils/string_utils.hpp"

#include <nlohmann/json_fwd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace pybind11 {
class gil_scoped_acquire;
}  // namespace pybind11

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

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

}  // namespace mrc::pymrc
