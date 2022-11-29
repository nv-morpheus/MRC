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

#include "pymrc/utilities/deserializers.hpp"
#include "pymrc/utilities/serializers.hpp"

#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <array>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;
using namespace pybind11::literals;

PYMRC_TEST_CLASS(Serializer);

class PymrcPickleableSimple
{
  public:
    PymrcPickleableSimple(std::string s, int i) : m_string_val(std::move(s)), m_int_val(i){};
    ~PymrcPickleableSimple() = default;

    const std::string& string_value() const
    {
        return m_string_val;
    };
    int int_value() const
    {
        return m_int_val;
    }

  private:
    int m_int_val{42};
    std::string m_string_val{"A string"};
};

PYBIND11_EMBEDDED_MODULE(pymrc_test_module, m)
{
    auto PymrcPickleableSimple_ = pybind11::class_<PymrcPickleableSimple>(m, "PymrcPickleableSimple");

    PymrcPickleableSimple_.def(py::init<std::string&, int>());
    PymrcPickleableSimple_.def("string_value", &PymrcPickleableSimple::string_value);
    PymrcPickleableSimple_.def("int_value", &PymrcPickleableSimple::int_value);
    PymrcPickleableSimple_.def(pybind11::pickle(
        [](const PymrcPickleableSimple& ptc) {  // __getstate__
            return pybind11::make_tuple(ptc.string_value(), ptc.int_value());
        },
        [](pybind11::tuple info) {  // __setstate__
            if (info.size() != 2)
            {
                throw std::runtime_error{"Invalid State"};
            }

            PymrcPickleableSimple ptc(info[0].cast<std::string>(), info[1].cast<int>());

            return ptc;
        }));
}

TEST_F(TestSerializer, SimpleObject)
{
    py::gil_scoped_acquire gil;

    pybind11::int_ int_obj(5);

    auto result  = pymrc::Serializer::serialize(int_obj, false);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(int_obj.equal(rebuilt));
}

TEST_F(TestSerializer, SimpleObjectShmem)
{
    py::gil_scoped_acquire gil;

    pybind11::int_ int_obj(5);

    auto result  = pymrc::Serializer::serialize(int_obj, true);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(int_obj.equal(rebuilt));
}

TEST_F(TestSerializer, NestedObject)
{
    py::gil_scoped_acquire gil;

    pybind11::int_ int_obj(5);
    pybind11::function func = pybind11::module_::import("os").attr("getuid");
    pybind11::dict py_dict("func"_a = func, "int"_a = int_obj);

    auto result  = pymrc::Serializer::serialize(py_dict, false);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(py_dict.equal(rebuilt));
}

TEST_F(TestSerializer, NestedObjectShmem)
{
    py::gil_scoped_acquire gil;

    pybind11::int_ int_obj(5);
    pybind11::function func = pybind11::module_::import("os").attr("getuid");
    pybind11::dict py_dict("func"_a = func, "int"_a = int_obj);

    auto result  = pymrc::Serializer::serialize(py_dict, true);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(py_dict.equal(rebuilt));
}

TEST_F(TestSerializer, Pybind11Simple)
{
    py::gil_scoped_acquire gil;

    auto test_mod          = py::module_::import("pymrc_test_module");
    auto simple_pickleable = test_mod.attr("PymrcPickleableSimple")("another string", 42);

    auto result  = pymrc::Serializer::serialize(simple_pickleable, false);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(simple_pickleable.attr("string_value")().equal(rebuilt.attr("string_value")()));
    ASSERT_TRUE(simple_pickleable.attr("int_value")().equal(rebuilt.attr("int_value")()));
}

TEST_F(TestSerializer, Pybind11SimpleShmem)
{
    py::gil_scoped_acquire gil;

    auto test_mod          = py::module_::import("pymrc_test_module");
    auto simple_pickleable = test_mod.attr("PymrcPickleableSimple")("another string", 42);

    auto result  = pymrc::Serializer::serialize(simple_pickleable, true);
    auto rebuilt = pymrc::Deserializer::deserialize(std::get<0>(result), std::get<1>(result));

    ASSERT_TRUE(simple_pickleable.attr("string_value")().equal(rebuilt.attr("string_value")()));
    ASSERT_TRUE(simple_pickleable.attr("int_value")().equal(rebuilt.attr("int_value")()));
}

TEST_F(TestSerializer, cuDFObject)
{
    pybind11::gil_scoped_acquire gil;

    py::module_ mod_cudf;
    try
    {
        mod_cudf = py::module_::import("cudf");
    } catch (...)
    {
        GTEST_SKIP() << "Pybind import of cuDF failed, skipping test.";
    }

    std::stringstream sstream;

    sstream << "FIELD1,FIELD2,FIELD3,FIELD4,FILED5,FIELD6";
    for (int i = 0; i < 1000; ++i)
    {
        sstream << "abc,1,2,10/1/1,4,end" << std::endl;
    }

    auto py_string = py::str(sstream.str());
    auto py_buffer = py::buffer(py::bytes(py_string));
    auto dataframe = mod_cudf.attr("read_csv")(py_buffer);

    auto df_buffer_info = pymrc::Serializer::serialize(dataframe, false);
    auto df_rebuilt     = pymrc::Deserializer::deserialize(std::get<0>(df_buffer_info), std::get<1>(df_buffer_info));
    ASSERT_TRUE(df_rebuilt.equal(dataframe));

    auto df_buffer_info_shmem = pymrc::Serializer::serialize(dataframe, true);
    auto df_rebuilt_shmem =
        pymrc::Deserializer::deserialize(std::get<0>(df_buffer_info_shmem), std::get<1>(df_buffer_info_shmem));
    ASSERT_TRUE(df_rebuilt_shmem.equal(dataframe));
}

TEST_F(TestSerializer, BadSerializeUnpicklable)
{
    pybind11::gil_scoped_acquire gil;

    pybind11::module_ mod = pybind11::module_::import("os");
    pybind11::dict py_dict("mod"_a = mod);

    EXPECT_THROW(pymrc::Serializer::serialize(py_dict, false), pybind11::error_already_set);
}

TEST_F(TestSerializer, BadDeserialize)
{
    pybind11::gil_scoped_acquire gil;
    char badbytes[] = "123456\0";  // NOLINT

    EXPECT_THROW(pymrc::Deserializer::deserialize(badbytes, 4), pybind11::error_already_set);
}
