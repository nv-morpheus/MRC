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

#include "pysrf/module_wrappers/pickle.hpp"
#include "pysrf/utilities/object_cache.hpp"

#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <string>  // IWYU pragma: keep

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"
// IWYU pragma: no_include "rx-includes.hpp"
#include <string>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;
using namespace pybind11::literals;

PYSRF_TEST_CLASS(PickleWrapper);

TEST_F(TestPickleWrapper, BasicPickle)
{
    py::gil_scoped_acquire gil;
    auto pkl = pysrf::PythonPickleInterface();

    auto dict  = py::dict("key1"_a = 10, "key2"_a = "test string");
    auto bytes = pkl.pickle(dict);

    ASSERT_TRUE(bytes.get_type().equal(py::bytes().get_type()));
}

TEST_F(TestPickleWrapper, BadPickle)
{
    py::gil_scoped_acquire gil;

    auto pkl        = pysrf::PythonPickleInterface();
    py::module_ mod = py::module_::import("os");
    py::dict py_dict("mod"_a = mod);

    EXPECT_THROW(pkl.pickle(py_dict), py::error_already_set);
}

TEST_F(TestPickleWrapper, BasicUnpickle)
{
    py::gil_scoped_acquire gil;
    auto pkl = pysrf::PythonPickleInterface();

    auto dict    = py::dict("key1"_a = 10, "key2"_a = "test string");
    auto bytes   = pkl.pickle(dict);
    auto rebuilt = pkl.unpickle(bytes);

    ASSERT_TRUE(rebuilt.get_type().equal(py::dict().get_type()));
    ASSERT_TRUE(rebuilt.equal(dict));
}

TEST_F(TestPickleWrapper, BadUnpickle)
{
    auto pkl = pysrf::PythonPickleInterface();

    char badbytes[] = "123456\0";  // NOLINT
    py::bytes bad_pybytes(badbytes, 6);
    EXPECT_THROW(pkl.unpickle(bad_pybytes), py::error_already_set);
}
