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

#include "test_pymrc.hpp"

#include "pymrc/forward.hpp"
#include "pymrc/utils.hpp"

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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;

// Create values too big to fit in int & float types to ensure we can pass
// long & double types to both nlohmann/json and python
constexpr long LONGVAL{UINT_MAX + 1L};
constexpr double DOUBLEEVAL{FLT_MAX + 1.0 + DBL_MIN};

PYMRC_TEST_CLASS(Utils);

TEST_F(TestUtils, ImportModuleObject)
{
    auto module_def = std::make_unique<py::module_::module_def>();
    auto m          = py::module_::create_extension_module("pymrc_unittest", nullptr, module_def.get());

    EXPECT_EQ(py::getattr(m, "loads", py::none()), py::none());

    pymrc::import_module_object(m, "json", "loads");
    EXPECT_NE(py::getattr(m, "loads"), py::none());

    py::dict d = m.attr("loads")(py::str(R"({"test":"this"})"));
    EXPECT_EQ(d["test"].cast<std::string>(), "this"s);
}

TEST_F(TestUtils, CastFromJson)
{
    nlohmann::json j = {{"this", {{"is", "a test"}}},
                        {"alphabet", {"a", "b", "c"}},
                        {"ncc", 1701},
                        {"cost", 47.47},
                        {"long val", LONGVAL},
                        {"double val", DOUBLEEVAL}};

    VLOG(10) << j.dump(4);
    auto pyobj = pymrc::cast_from_json(j);

    EXPECT_EQ(pyobj["ncc"].cast<int>(), 1701);
    EXPECT_EQ(pyobj["cost"].cast<float>(), 47.47F);
    EXPECT_EQ(pyobj["long val"].cast<long>(), LONGVAL);
    EXPECT_EQ(pyobj["double val"].cast<double>(), DOUBLEEVAL);

    {
        std::vector<std::string> expected{"a", "b", "c"};
        EXPECT_EQ(pyobj["alphabet"].cast<std::vector<std::string>>(), expected);
    }
    {
        using expected_t = std::array<std::string, 3>;
        expected_t expected{"a"s, "b"s, "c"s};
        EXPECT_EQ(pyobj["alphabet"].cast<expected_t>(), expected);
    }

    {
        using expected_t = std::map<std::string, std::string>;
        expected_t expected{{"is", "a test"}};
        EXPECT_EQ(pyobj["this"].cast<expected_t>(), expected);
    }
}

TEST_F(TestUtils, CastFromPyObject)
{
    {
        py::dict d;
        d[py::str("test"s)] = py::str("this"s);

        auto j = pymrc::cast_from_pyobject(d);

        EXPECT_EQ(j.dump(), "{\"test\":\"this\"}"s);
    }

    {
        py::object json = py::module_::import("json");
        py::dict d =
            json.attr("loads")(py::str("{\"this\": {\"is\":\"a test\"},"s
                                       " \"alphabet\": [\"a\", \"b\", \"c\"],"s
                                       " \"ncc\": 1701,"s
                                       " \"cost\": 47.47,"s
                                       " \"long val\": 4294967296,"s
                                       " \"double val\": 3.4028234663852886e+38}"s));

        auto j = pymrc::cast_from_pyobject(d);

        EXPECT_EQ(j["ncc"].get<int>(), 1701);
        EXPECT_EQ(j["cost"].get<float>(), 47.47F);
        EXPECT_EQ(j["long val"].get<long>(), LONGVAL);
        EXPECT_EQ(j["double val"].get<double>(), DOUBLEEVAL);

        {
            std::vector<std::string> expected{"a", "b", "c"};
            EXPECT_EQ(j["alphabet"].get<std::vector<std::string>>(), expected);
        }
        {
            using expected_t = std::array<std::string, 3>;
            expected_t expected{"a"s, "b"s, "c"s};
            EXPECT_EQ(j["alphabet"].get<expected_t>(), expected);
        }

        {
            using expected_t = std::map<std::string, std::string>;
            expected_t expected{{"is", "a test"}};
            EXPECT_EQ(j["this"].get<expected_t>(), expected);
        }
    }
}

TEST_F(TestUtils, PyObjectWrapper)
{
    py::list test_list;

    EXPECT_EQ(test_list.ref_count(), 1);

    mrc::pymrc::PyObjectWrapper wrapper(std::move(test_list));

    // Initial object should be empty
    EXPECT_FALSE(test_list);

    // Internal object reference count should be 1
    EXPECT_EQ(wrapper.view_obj().ref_count(), 1);

    // Move the wrapper into a new object
    py::object test_obj = std::move(wrapper);

    // Wrapper shouldnt hold anything anymore
    EXPECT_FALSE(wrapper.view_obj());

    // Ref count of wrapper should still be 1
    EXPECT_EQ(test_obj.ref_count(), 1);

    // Create by assignment
    wrapper = py::list();

    // Should now hold an object
    EXPECT_TRUE(wrapper.view_obj());
    EXPECT_EQ(wrapper.view_obj().ref_count(), 1);
}

TEST_F(TestUtils, PyObjectHolder)
{
    py::list test_list;

    EXPECT_EQ(test_list.ref_count(), 1);

    mrc::pymrc::PyObjectHolder wrapper(std::move(test_list));

    // Initial object should be empty
    EXPECT_FALSE(test_list);

    // Internal object reference count should be 1
    EXPECT_EQ(wrapper.view_obj().ref_count(), 1);

    // Move the wrapper into a new object
    py::object test_obj = std::move(wrapper);

    // Wrapper shouldnt hold anything anymore
    EXPECT_FALSE(wrapper.view_obj());

    // Ref count of wrapper should still be 1
    EXPECT_EQ(test_obj.ref_count(), 1);

    // Create by assignment
    wrapper = py::list();

    // Should now hold an object
    EXPECT_TRUE(wrapper.view_obj());
    EXPECT_EQ(wrapper.view_obj().ref_count(), 1);

    // Make a copy of the holder
    mrc::pymrc::PyObjectHolder holder_copy = wrapper;

    // Ref count should be the same
    EXPECT_TRUE(wrapper.view_obj());
    EXPECT_EQ(wrapper.view_obj().ref_count(), 1);
}
