/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/fiber/future/promise.hpp>

#include <future>
#include <optional>
#include <string>

namespace pybind11 {
class object;
}  // namespace pybind11

namespace mrc::pymrc {
struct OnDataFunction;
template <typename SignatureT>
struct PyFuncHolder;

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class PythonOperator
{
  public:
    PythonOperator(std::string name, PyObjectOperateFn operate_fn);

    const std::string& get_name() const;

    const PyObjectOperateFn& get_operate_fn() const;

  private:
    std::string m_name;
    PyObjectOperateFn m_operate_fn;
};

class OperatorProxy
{
  public:
    static std::string get_name(PythonOperator& self);
};

class AsyncOperatorHandler
{
  public:
    AsyncOperatorHandler();
    ~AsyncOperatorHandler() = default;

    boost::fibers::future<PyObjectHolder> process_async_generator(PyObjectHolder asyncgen);

  private:
    pybind11::module_ m_asyncio;
};

class OperatorsProxy
{
  public:
    static PythonOperator build(PyFuncHolder<void(const PyObjectObservable& obs, PyObjectSubscriber& sub)> build_fn);
    static PythonOperator filter(PyFuncHolder<bool(pybind11::object x)> filter_fn);
    static PythonOperator flatten();
    static PythonOperator concat_map_async(PyFuncHolder<PyObjectHolder(pybind11::object)> flatmap_fn);
    static PythonOperator map(OnDataFunction map_fn);
    static PythonOperator on_completed(PyFuncHolder<std::optional<pybind11::object>()> finally_fn);
    static PythonOperator pairwise();
    static PythonOperator to_list();
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
