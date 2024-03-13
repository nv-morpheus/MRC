/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_pymrc.hpp"

#include "pymrc/utilities/json_values.hpp"
#include "pymrc/utils.hpp"  // for imort_module_object

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <array>
#include <cfloat>
#include <climits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace mrc::pymrc;
using namespace std::string_literals;
using namespace pybind11::literals;  // to bring in the `_a` literal

PYMRC_TEST_CLASS(JSONValues);

TEST_F(TestJSONValues, ToPythonSerializable)
{
    py::dict py_dict;
    py_dict[py::str("test"s)] = py::str("this"s);

    JSONValues j{py_dict};
    auto result = j.to_python();

    EXPECT_EQ(result, py_dict);
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the object and not stored it
}

TEST_F(TestJSONValues, ToPythonRootUnserializable)
{
    py::object py_dec = py::module_::import("decimal").attr("Decimal")("1.0");

    JSONValues j{py_dec};
    auto result = j.to_python();

    EXPECT_TRUE(result.is(py_dec));  // Ensure we stored the object
}

TEST_F(TestJSONValues, ToPythonNestedDictUnserializable)
{
    // decimal.Decimal is not serializable
    py::object Decimal = py::module_::import("decimal").attr("Decimal");

    py::object py_dec1 = Decimal("1.1");
    py::object py_dec2 = Decimal("1.2");
    py::object py_dec3 = Decimal("1.3");

    std::vector<py::object> py_values = {py::cast(1), py::cast(2), py_dec3, py::cast(4)};
    py::list py_list                  = py::cast(py_values);

    // Test with object in a nested dict
    py::dict py_dict("a"_a           = py::dict("b"_a = py::dict("c"_a = py::dict("d"_a = py_dec1))),
                     "other"_a       = py_dec2,
                     "nested_list"_a = py_list);

    JSONValues j{py_dict};
    auto result = j.to_python();
    EXPECT_EQ(result, py_dict);
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the object and not stored it

    // Individual Decimal instances shoudl be stored and thus pass an `is` test
    py::object result_dec1 = result["a"]["b"]["c"]["d"];
    EXPECT_TRUE(result_dec1.is(py_dec1));

    py::object result_dec2 = result["other"];
    EXPECT_TRUE(result_dec2.is(py_dec2));

    py::list nested_list   = result["nested_list"];
    py::object result_dec3 = nested_list[2];
    EXPECT_TRUE(result_dec3.is(py_dec3));
}

TEST_F(TestJSONValues, ToPythonList) {}
