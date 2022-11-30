/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pytypes.h>

#include <functional>
#include <string>
#include <utility>

// IWYU pragma: no_include "pymrc/node.hpp"

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class PythonOperator
{
  public:
    PythonOperator(std::string name, PyObjectOperateFn operate_fn) :
      m_name(std::move(name)),
      m_operate_fn(std::move(operate_fn))
    {}

    const std::string& get_name() const
    {
        return m_name;
    }

    const PyObjectOperateFn& get_operate_fn() const
    {
        return m_operate_fn;
    }

  private:
    std::string m_name;
    PyObjectOperateFn m_operate_fn;
};

class OperatorProxy
{
  public:
    static std::string get_name(PythonOperator& self);
};

class OperatorsProxy
{
  public:
    static PythonOperator filter(std::function<bool(pybind11::object x)> filter_fn);
    static PythonOperator flatten();
    static PythonOperator map(std::function<pybind11::object(pybind11::object x)> map_fn);
    static PythonOperator on_completed(std::function<pybind11::object()> finally_fn);
    static PythonOperator pairwise();
    static PythonOperator to_list();
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
