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

#include "pymrc/types.hpp"
#include "pymrc/utilities/json_values.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <array>
#include <cstddef>           // for size_t
#include <initializer_list>  // for initializer_list
#include <stdexcept>
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

py::dict mk_py_dict()
{
    // return a simple python dict with a nested dict, a list, an integer, and a float
    std::array<std::string, 3> alphabet = {"a", "b", "c"};
    return py::dict("this"_a     = py::dict("is"_a = "a test"s),
                    "alphabet"_a = py::cast(alphabet),
                    "ncc"_a      = 1701,
                    "cost"_a     = 47.47);
}

nlohmann::json mk_json()
{
    // return a simple json object comparable to that returned by mk_py_dict
    return {{"this", {{"is", "a test"}}}, {"alphabet", {"a", "b", "c"}}, {"ncc", 1701}, {"cost", 47.47}};
}

py::object mk_decimal(const std::string& value = "1.0"s)
{
    // return a Python decimal.Decimal object, as a simple object without a supported JSON serialization
    return py::module_::import("decimal").attr("Decimal")(value);
}

TEST_F(TestJSONValues, DefaultConstructor)
{
    JSONValues j;

    EXPECT_EQ(j.to_json(JSONValues::stringify), nlohmann::json());
    EXPECT_TRUE(j.to_python().is_none());
}

TEST_F(TestJSONValues, ToPythonSerializable)
{
    auto py_dict = mk_py_dict();

    JSONValues j{py_dict};
    auto result = j.to_python();

    EXPECT_TRUE(result.equal(py_dict));
    EXPECT_FALSE(result.is(py_dict));  // Ensure we actually serialized the object and not stored it
}

TEST_F(TestJSONValues, ToPythonFromJSON)
{
    py::dict py_expected_results = mk_py_dict();

    nlohmann::json json_input = mk_json();
    JSONValues j{json_input};
    auto result = j.to_python();

    EXPECT_TRUE(result.equal(py_expected_results));
}

TEST_F(TestJSONValues, ToJSONFromPython)
{
    auto expected_results = mk_json();

    py::dict py_input = mk_py_dict();

    JSONValues j{py_input};
    auto result = j.to_json(JSONValues::stringify);

    EXPECT_EQ(result, expected_results);
}

TEST_F(TestJSONValues, ToJSONFromPythonUnserializable)
{
    std::string dec_val{"2.2"};
    auto expected_results     = mk_json();
    expected_results["other"] = dec_val;

    py::dict py_input = mk_py_dict();
    py_input["other"] = mk_decimal(dec_val);

    JSONValues j{py_input};
    EXPECT_EQ(j.to_json(JSONValues::stringify), expected_results);
}

TEST_F(TestJSONValues, ToJSONFromJSON)
{
    JSONValues j{mk_json()};
    auto result = j.to_json(JSONValues::stringify);

    EXPECT_EQ(result, mk_json());
}

TEST_F(TestJSONValues, ToPythonRootUnserializable)
{
    py::object py_dec = mk_decimal();

    JSONValues j{py_dec};
    auto result = j.to_python();

    EXPECT_TRUE(result.equal(py_dec));
    EXPECT_TRUE(result.is(py_dec));  // Ensure we stored the object

    nlohmann::json expexted_json("**pymrc_placeholder"s);
    EXPECT_EQ(j.view_json(), expexted_json);
}

TEST_F(TestJSONValues, ToPythonSimpleDict)
{
    py::object py_dec = mk_decimal();
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
    py::object py_dec1 = mk_decimal("1.1");
    py::object py_dec2 = mk_decimal("1.2");
    py::object py_dec3 = mk_decimal("1.3");

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
    py::object py_dec = mk_decimal("1.1"s);

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

TEST_F(TestJSONValues, NumUnserializable)
{
    {
        JSONValues j{mk_json()};
        EXPECT_EQ(j.num_unserializable(), 0);
        EXPECT_FALSE(j.has_unserializable());
    }
    {
        JSONValues j{mk_py_dict()};
        EXPECT_EQ(j.num_unserializable(), 0);
        EXPECT_FALSE(j.has_unserializable());
    }
    {
        // Test with object in a nested dict
        py::object py_dec = mk_decimal();
        {
            py::dict d("a"_a = py::dict("b"_a = py::dict("c"_a = py::dict("d"_a = py_dec))), "other"_a = 2);

            JSONValues j{d};
            EXPECT_EQ(j.num_unserializable(), 1);
            EXPECT_TRUE(j.has_unserializable());
        }
        {
            // Storing the same object twice should count twice
            py::dict d("a"_a = py::dict("b"_a = py::dict("c"_a = py::dict("d"_a = py_dec))), "other"_a = py_dec);

            JSONValues j{d};
            EXPECT_EQ(j.num_unserializable(), 2);
            EXPECT_TRUE(j.has_unserializable());
        }
        {
            py::object py_dec2 = mk_decimal("2.0");
            py::dict d("a"_a     = py::dict("b"_a = py::dict("c"_a = py::dict("d"_a = py_dec, "e"_a = py_dec2))),
                       "other"_a = py_dec);

            JSONValues j{d};
            EXPECT_EQ(j.num_unserializable(), 3);
            EXPECT_TRUE(j.has_unserializable());
        }
    }
}

TEST_F(TestJSONValues, SetValueNewKeyJSON)
{
    // Set to new key that doesn't exist
    auto expected_results     = mk_json();
    expected_results["other"] = mk_json();

    JSONValues values{mk_json()};
    auto new_values = values.set_value("/other", mk_json());
    EXPECT_EQ(new_values.to_json(JSONValues::stringify), expected_results);
}

TEST_F(TestJSONValues, SetValueExistingKeyJSON)
{
    // Set to existing key
    auto expected_results    = mk_json();
    expected_results["this"] = mk_json();

    JSONValues values{mk_json()};
    auto new_values = values.set_value("/this", mk_json());
    EXPECT_EQ(new_values.to_json(JSONValues::stringify), expected_results);
}

TEST_F(TestJSONValues, SetValueNewKeyJSONWithUnserializable)
{
    // Set to new key that doesn't exist
    auto expected_results     = mk_py_dict();
    expected_results["other"] = mk_py_dict();
    expected_results["dec"]   = mk_decimal();

    auto input   = mk_py_dict();
    input["dec"] = mk_decimal();

    JSONValues values{input};
    auto new_values = values.set_value("/other", mk_json());
    EXPECT_TRUE(new_values.to_python().equal(expected_results));
}

TEST_F(TestJSONValues, SetValueExistingKeyJSONWithUnserializable)
{
    // Set to existing key
    auto expected_results    = mk_py_dict();
    expected_results["dec"]  = mk_decimal();
    expected_results["this"] = mk_py_dict();

    auto input   = mk_py_dict();
    input["dec"] = mk_decimal();

    JSONValues values{input};
    auto new_values = values.set_value("/this", mk_json());
    EXPECT_TRUE(new_values.to_python().equal(expected_results));
}

TEST_F(TestJSONValues, SetValueNewKeyPython)
{
    // Set to new key that doesn't exist
    auto expected_results     = mk_py_dict();
    expected_results["other"] = mk_decimal();

    JSONValues values{mk_json()};
    auto new_values = values.set_value("/other", mk_decimal());
    EXPECT_TRUE(new_values.to_python().equal(expected_results));
}

TEST_F(TestJSONValues, SetValueNestedUnsupportedPython)
{
    JSONValues values{mk_json()};
    EXPECT_THROW(values.set_value("/other/nested", mk_decimal()), py::error_already_set);
}

TEST_F(TestJSONValues, SetValueNestedUnsupportedJSON)
{
    JSONValues values{mk_json()};
    EXPECT_THROW(values.set_value("/other/nested", nlohmann::json(1.0)), nlohmann::json::out_of_range);
}

TEST_F(TestJSONValues, SetValueExistingKeyPython)
{
    // Set to existing key
    auto expected_results    = mk_py_dict();
    expected_results["this"] = mk_decimal();

    JSONValues values{mk_json()};
    auto new_values = values.set_value("/this", mk_decimal());
    EXPECT_TRUE(new_values.to_python().equal(expected_results));
}

TEST_F(TestJSONValues, SetValueNewKeyJSONDefaultConstructed)
{
    nlohmann::json expected_results{{"other", mk_json()}};

    JSONValues values;
    auto new_values = values.set_value("/other", mk_json());
    EXPECT_EQ(new_values.to_json(JSONValues::stringify), expected_results);
}

TEST_F(TestJSONValues, SetValueJSONValues)
{
    // Set to new key that doesn't exist
    auto expected_results     = mk_json();
    expected_results["other"] = mk_json();

    JSONValues values1{mk_json()};
    JSONValues values2{mk_json()};
    auto new_values = values1.set_value("/other", values2);
    EXPECT_EQ(new_values.to_json(JSONValues::stringify), expected_results);
}

TEST_F(TestJSONValues, SetValueJSONValuesWithUnserializable)
{
    // Set to new key that doesn't exist
    auto expected_results     = mk_py_dict();
    expected_results["other"] = py::dict("dec"_a = mk_decimal());

    JSONValues values1{mk_json()};

    auto input_dict = py::dict("dec"_a = mk_decimal());
    JSONValues values2{input_dict};

    auto new_values = values1.set_value("/other", values2);
    EXPECT_TRUE(new_values.to_python().equal(expected_results));
}

TEST_F(TestJSONValues, GetJSON)
{
    using namespace nlohmann;
    const auto json_doc            = mk_json();
    std::vector<std::string> paths = {"/", "/this", "/this/is", "/alphabet", "/ncc", "/cost"};
    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& path : paths)
        {
            json::json_pointer jp;
            if (path != "/")
            {
                jp = json::json_pointer(path);
            }

            EXPECT_TRUE(json_doc.contains(jp)) << "Path: '" << path << "' not found in json";
            EXPECT_EQ(value.get_json(path, JSONValues::stringify), json_doc[jp]);
        }
    }
}

TEST_F(TestJSONValues, GetJSONError)
{
    std::vector<std::string> paths = {"/doesntexist", "/this/fake"};
    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& path : paths)
        {
            EXPECT_THROW(value.get_json(path, JSONValues::stringify), std::runtime_error);
        }
    }
}

TEST_F(TestJSONValues, GetPython)
{
    const auto py_dict = mk_py_dict();

    // <path, expected_result>
    std::vector<std::pair<std::string, py::object>> tests = {{"/", py_dict},
                                                             {"/this", py::dict("is"_a = "a test"s)},
                                                             {"/this/is", py::str("a test"s)},
                                                             {"/alphabet", py_dict["alphabet"]},
                                                             {"/ncc", py::int_(1701)},
                                                             {"/cost", py::float_(47.47)}};

    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& p : tests)
        {
            const auto& path            = p.first;
            const auto& expected_result = p.second;
            EXPECT_TRUE(value.get_python(path).equal(expected_result));
        }
    }
}

TEST_F(TestJSONValues, GetPythonError)
{
    std::vector<std::string> paths = {"/doesntexist", "/this/fake"};
    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& path : paths)
        {
            EXPECT_THROW(value.get_python(path), std::runtime_error) << "Expected failure with path: '" << path << "'";
        }
    }
}

TEST_F(TestJSONValues, SubscriptOpt)
{
    using namespace nlohmann;
    const auto json_doc             = mk_json();
    std::vector<std::string> values = {"", "this", "this/is", "alphabet", "ncc", "cost"};
    std::vector<std::string> paths;
    for (const auto& value : values)
    {
        paths.push_back(value);
        paths.push_back("/" + value);
    }

    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& path : paths)
        {
            auto jv = value[path];

            json::json_pointer jp;
            if (!path.empty() && path != "/")
            {
                std::string json_path = path;
                if (json_path[0] != '/')
                {
                    json_path = "/"s + json_path;
                }

                jp = json::json_pointer(json_path);
            }

            EXPECT_EQ(jv.to_json(JSONValues::stringify), json_doc[jp]);
        }
    }
}

TEST_F(TestJSONValues, SubscriptOptError)
{
    std::vector<std::string> paths = {"/doesntexist", "/this/fake"};
    for (const auto& value : {JSONValues{mk_json()}, JSONValues{mk_py_dict()}})
    {
        for (const auto& path : paths)
        {
            EXPECT_THROW(value[path], std::runtime_error);
        }
    }
}

TEST_F(TestJSONValues, Stringify)
{
    auto dec_val = mk_decimal("2.2"s);
    EXPECT_EQ(JSONValues::stringify(dec_val, "/"s), nlohmann::json("2.2"s));
}

TEST_F(TestJSONValues, CastPyToJSONValues)
{
    auto py_dict = mk_py_dict();
    auto j       = py_dict.cast<JSONValues>();
    EXPECT_TRUE(j.to_python().equal(py_dict));
}

TEST_F(TestJSONValues, CastJSONValuesToPy)
{
    auto j       = JSONValues{mk_json()};
    auto py_dict = py::cast(j);
    EXPECT_TRUE(py_dict.equal(j.to_python()));
}
