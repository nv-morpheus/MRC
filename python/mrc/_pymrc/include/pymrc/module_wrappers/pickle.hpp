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

#include "pymrc/utilities/object_cache.hpp"

#include <pybind11/pytypes.h>

namespace mrc::pymrc {
#pragma GCC visibility push(default)
/****** PythonPickleInterface****************************************/
/**
 * @brief Light wrapper around the python pickle module.
 */

class PythonPickleInterface
{
  public:
    ~PythonPickleInterface();
    PythonPickleInterface();

    pybind11::bytes pickle(pybind11::object obj);
    pybind11::object unpickle(pybind11::bytes bytes);

  private:
    PythonObjectCache& m_pycache;

    pybind11::function m_func_loads{};
    pybind11::function m_func_dumps{};
};
#pragma GCC visibility pop
}  // namespace mrc::pymrc
