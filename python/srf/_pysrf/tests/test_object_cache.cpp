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

#include "test_pysrf.hpp"

#include "pysrf/utilities/object_cache.hpp"

#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // IWYU pragma: keep
#include <string>   // IWYU pragma: keep

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"
// IWYU pragma: no_include "rx-includes.hpp"
#include <cstddef>
#include <string>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;
using namespace pybind11::literals;

PYSRF_TEST_CLASS(ObjectCache);

TEST_F(TestObjectCache, Acquire)
{
    srf::pysrf::PythonObjectCache& cache = pysrf::PythonObjectCache::get_handle();
}

TEST_F(TestObjectCache, Interface)
{
    srf::pysrf::PythonObjectCache& cache = pysrf::PythonObjectCache::get_handle();

    auto sys = cache.get_module("sys");
    auto os  = cache.get_module("os");
    ASSERT_TRUE(cache.size() == 2);

    std::size_t os_ref = os.ref_count();

    auto os2 = cache.get_module("os");
    ASSERT_TRUE(cache.size() == 2);
    ASSERT_TRUE(os.ref_count() > os_ref);

    ASSERT_TRUE(cache.contains("sys"));

    py::object obj = py::dict("key"_a = "val");
    cache.cache_object("test_dictionary", obj);
    ASSERT_TRUE(cache.size() == 3);
    ASSERT_TRUE(cache.contains("test_dictionary"));

    auto regex = cache.get_or_load("re", []() { return py::module_::import("re"); });
    ASSERT_TRUE(cache.contains("re"));

    EXPECT_THROW(cache.get_module("not_a_real_module"), py::error_already_set);
}
