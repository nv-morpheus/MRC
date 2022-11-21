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

#include "pysrf/codable_object.hpp"  // IWYU pragma: keep
#include "pysrf/forward.hpp"
#include "pysrf/types.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/decode.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/codable/encoding_options.hpp"
#include "srf/codable/type_traits.hpp"

#include <gtest/gtest.h>
#include <object.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <string>  // IWYU pragma: keep

// IWYU pragma: no_include <gtest/gtest-message.h>
// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <gtest/gtest-test-part.h>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <tupleobject.h>
#include <string>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;
using namespace srf::codable;
using namespace pybind11::literals;

PYSRF_TEST_CLASS(CodablePyobject);

TEST_F(TestCodablePyobject, PyObject)
{
    static_assert(is_codable_v<pysrf::PyHolder>, "pybind11::object should be codable.");
    static_assert(is_codable_v<py::object>, "pybind11::object should be codable.");
    static_assert(is_decodable_v<py::object>, "pybind11::object should be decodable.");
    static_assert(is_decodable_v<pysrf::PyHolder>, "pybind11::object should be decodable.");
    static_assert(!is_codable_v<PyObject>,
                  "No support for directly coding cpython objects -- "
                  "use pybind11::object or srf::PyHolder");
    static_assert(!is_decodable_v<PyObject>,
                  "No support for directly coding cpython objects -- "
                  "use pybind11::object or srf::PyHolder");
}

TEST_F(TestCodablePyobject, EncodedObjectSimple)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    py::object py_dict = py::dict("prop1"_a = py::none(),
                                  "prop2"_a = 123,
                                  "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                                  "prop4"_a = py::bool_(false),
                                  "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    py::dict py_dict_deserialized = decode<py::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedHolderObjectSimple)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    pysrf::PyHolder py_dict = py::dict("prop1"_a = py::none(),
                                       "prop2"_a = 123,
                                       "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                                       "prop4"_a = py::bool_(false),
                                       "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode(py_dict, enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pysrf::PyHolder py_dict_deserialized = decode<pysrf::PyHolder>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedObjectSharedMem)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    py::dict py_dict("prop1"_a = py::none(),
                     "prop2"_a = 123,
                     "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                     "prop4"_a = py::bool_(false),
                     "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, true);
    encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    py::dict py_dict_deserialized = decode<py::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedHolderObjectSharedMem)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    pysrf::PyHolder py_dict = py::dict("prop1"_a = py::none(),
                                       "prop2"_a = 123,
                                       "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                                       "prop4"_a = py::bool_(false),
                                       "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, true);
    encode(py_dict, enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pysrf::PyHolder py_dict_deserialized = decode<pysrf::PyHolder>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedObjectSharedMemNoCopy)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    py::dict py_dict("prop1"_a = py::none(),
                     "prop2"_a = 123,
                     "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                     "prop4"_a = py::bool_(false),
                     "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    py::dict py_dict_deserialized = decode<py::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedHolderObjectSharedMemNoCopy)
{
    py::gil_scoped_acquire gil;

    py::module_ mod = py::module_::import("os");

    pysrf::PyHolder py_dict = py::dict("prop1"_a = py::none(),
                                       "prop2"_a = 123,
                                       "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
                                       "prop4"_a = py::bool_(false),
                                       "func"_a  = py::getattr(mod, "getuid"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode(py_dict, enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pysrf::PyHolder py_dict_deserialized = decode<pysrf::PyHolder>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, BadUnpickleable)
{
    py::gil_scoped_acquire gil;
    py::dict py_dict("mod(unpickleable)"_a = py::module_::import("sys"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);

    EXPECT_THROW(encode(py::cast<py::object>(py_dict), enc_obj, enc_ops), py::error_already_set);
}

TEST_F(TestCodablePyobject, BadHolderUnpickleable)
{
    py::gil_scoped_acquire gil;
    pysrf::PyHolder py_dict = py::dict("mod(unpickleable)"_a = py::module_::import("sys"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);

    EXPECT_THROW(encode(py_dict, enc_obj, enc_ops), py::error_already_set);
}
