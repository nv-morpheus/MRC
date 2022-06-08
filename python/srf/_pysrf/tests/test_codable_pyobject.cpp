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

#include <pysrf/utilities/deserializers.hpp>
#include <pysrf/utilities/serializers.hpp>
#include <pysrf/codable_object.hpp>

#include <srf/codable/decode.hpp>
#include <srf/codable/encode.hpp>
#include <srf/codable/encoded_object.hpp>
#include <srf/codable/encoding_options.hpp>
#include <srf/codable/type_traits.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <vector>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;
using namespace srf::codable;
using namespace pybind11::literals;

PYSRF_TEST_CLASS(CodablePyobject);

TEST_F(TestCodablePyobject, PyObject) {
    static_assert(is_codable_v<py::object>, "pybind11::object should be codable.");
    static_assert(is_decodable_v<py::object>, "pybind11::object should be decodable.");
    static_assert(!is_codable_v<PyObject>, "No support for directly coding cpython objects -- use pybind11::object");
    static_assert(!is_decodable_v<PyObject>, "No support for directly coding cpython objects -- use pybind11::object");
}

TEST_F(TestCodablePyobject, EncodedObjectSimple) {
    pybind11::gil_scoped_acquire gil;

    pybind11::module_ mod = pybind11::module_::import("os");

    pybind11::dict py_dict("prop1"_a=pybind11::none(),
                           "prop2"_a=123,
                           "prop3"_a=pybind11::dict("subprop1"_a=1,
                                                      "subprop2"_a="abc"),
                           "prop4"_a=pybind11::bool_(false),
                           "func"_a=pybind11::getattr(mod, "getuid")
    );

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode (pybind11::cast<pybind11::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pybind11::dict py_dict_deserialized = decode<pybind11::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedObjectSharedMem) {
    pybind11::gil_scoped_acquire gil;

    pybind11::module_ mod = pybind11::module_::import("os");

    pybind11::dict py_dict("prop1"_a=pybind11::none(),
                           "prop2"_a=123,
                           "prop3"_a=pybind11::dict("subprop1"_a=1,
                                                      "subprop2"_a="abc"),
                           "prop4"_a=pybind11::bool_(false),
                           "func"_a=pybind11::getattr(mod, "getuid")
    );

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, true);
    encode (pybind11::cast<pybind11::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pybind11::dict py_dict_deserialized = decode<pybind11::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, EncodedObjectSharedMemNoCopy) {
    pybind11::gil_scoped_acquire gil;

    pybind11::module_ mod = pybind11::module_::import("os");

    pybind11::dict py_dict("prop1"_a=pybind11::none(),
                           "prop2"_a=123,
                           "prop3"_a=pybind11::dict("subprop1"_a=1,
                                                      "subprop2"_a="abc"),
                           "prop4"_a=pybind11::bool_(false),
                           "func"_a=pybind11::getattr(mod, "getuid")
    );

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);
    encode (pybind11::cast<pybind11::object>(py_dict), enc_obj, enc_ops);

    EXPECT_EQ(enc_obj.object_count(), 1);
    EXPECT_EQ(enc_obj.descriptor_count(), 1);

    pybind11::dict py_dict_deserialized = decode<pybind11::object>(enc_obj);

    EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
    EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
    EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
}

TEST_F(TestCodablePyobject, BadUnpickleable) {
    pybind11::gil_scoped_acquire gil;
    pybind11::dict py_dict("mod(unpickleable)"_a=pybind11::module_::import("sys"));

    EncodedObject enc_obj;
    EncodingOptions enc_ops(true, false);

    EXPECT_THROW(
        encode (pybind11::cast<pybind11::object>(py_dict), enc_obj, enc_ops),
        py::error_already_set
    );
}

