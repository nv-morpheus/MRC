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

#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // for size_t
#include <string>
#include <utility>  // for pair
#include <vector>
// We already included pybind11.h don't need these others
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <pybind11/eval.h>
// IWYU pragma: no_include <pybind11/pytypes.h>

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

    EXPECT_TRUE(result.equal(py_dict));
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the object and not stored it
}

TEST_F(TestJSONValues, ToPythonRootUnserializable)
{
    py::object py_dec = py::module_::import("decimal").attr("Decimal")("1.0");

    JSONValues j{py_dec};
    auto result = j.to_python();

    EXPECT_TRUE(result.equal(py_dec));
    EXPECT_TRUE(result.is(py_dec));  // Ensure we stored the object
}

TEST_F(TestJSONValues, ToPythonSimpleDict)
{
    py::object py_dec = py::module_::import("decimal").attr("Decimal")("1.0");
    py::dict py_dict;
    py_dict[py::str("test"s)] = py_dec;

    JSONValues j{py_dict};
    py::dict result = j.to_python();

    EXPECT_TRUE(result.equal(py_dict));
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the dict and not stored it

    py::object result_dec = result["test"];
    EXPECT_TRUE(result_dec.is(py_dec));  // Ensure we stored the decimal object
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
    EXPECT_TRUE(result.equal(py_dict));
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

TEST_F(TestJSONValues, ToPythonList)
{
    py::object Decimal = py::module_::import("decimal").attr("Decimal");
    py::object py_dec  = Decimal("1.1");

    std::vector<py::object> py_values = {py::cast(1), py::cast(2), py_dec, py::cast(4)};
    py::list py_list                  = py::cast(py_values);

    JSONValues j{py_list};
    py::list result = j.to_python();
    EXPECT_TRUE(result.equal(py_list));
    py::object result_dec = result[2];
    EXPECT_TRUE(result_dec.is(py_dec));
}

TEST_F(TestJSONValues, ToPythonMultipleTypes)
{
    // Test with miultiple types not json serializable: module, class, function, generator
    py::object py_mod  = py::module_::import("decimal");
    py::object py_cls  = py_mod.attr("Decimal");
    py::object globals = py::globals();
    py::exec(
        R"(
            def gen_fn():
                yield 1
        )",
        globals);

    py::object py_fn  = globals["gen_fn"];
    py::object py_gen = py_fn();

    std::vector<std::pair<std::size_t, py::object>> expected_list_objs = {{1, py_mod},
                                                                          {3, py_cls},
                                                                          {5, py_fn},
                                                                          {7, py_gen}};

    std::vector<py::object> py_values =
        {py::cast(0), py_mod, py::cast(2), py_cls, py::cast(4), py_fn, py::cast(6), py_gen};
    py::list py_list = py::cast(py_values);

    std::vector<std::pair<std::string, py::object>> expected_dict_objs = {{"module", py_mod},
                                                                          {"class", py_cls},
                                                                          {"function", py_fn},
                                                                          {"generator", py_gen}};

    // Test with object in a nested dict
    py::dict py_dict("module"_a      = py_mod,
                     "class"_a       = py_cls,
                     "function"_a    = py_fn,
                     "generator"_a   = py_gen,
                     "nested_list"_a = py_list);

    JSONValues j{py_dict};
    auto result = j.to_python();
    EXPECT_TRUE(result.equal(py_dict));
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the object and not stored it

    for (const auto& [key, value] : expected_dict_objs)
    {
        py::object result_value = result[key.c_str()];
        EXPECT_TRUE(result_value.is(value));
    }

    py::list nested_list = result["nested_list"];
    for (const auto& [index, value] : expected_list_objs)
    {
        py::object result_value = nested_list[index];
        EXPECT_TRUE(result_value.is(value));
    }
}
