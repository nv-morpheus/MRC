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

// iwyu headers for the narrowed tests

#include "test_pymrc.hpp"

#include "pymrc/codable_object.hpp"  // IWYU pragma: keep
#include "pymrc/forward.hpp"
#include "pymrc/types.hpp"

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/type_traits.hpp"

#include <gtest/gtest.h>
#include <object.h>
#include <pybind11/pytypes.h>

#include <string>  // IWYU pragma: keep

// uncomment the following header list when uncommenting the test

// #include "test_pymrc.hpp"

// #include "pymrc/codable_object.hpp"  // IWYU pragma: keep
// #include "pymrc/forward.hpp"
// #include "pymrc/types.hpp"

// #include "mrc/codable/api.hpp"
// #include "mrc/codable/codable_protocol.hpp"
// #include "mrc/codable/decode.hpp"
// #include "mrc/codable/encode.hpp"
// #include "mrc/codable/encoded_object.hpp"
// #include "mrc/codable/encoding_options.hpp"
// #include "mrc/codable/storage_forwarder.hpp"
// #include "mrc/codable/type_traits.hpp"
// #include "mrc/protos/codable.pb.h"

// #include <gtest/gtest.h>
// #include <object.h>
// #include <pybind11/cast.h>
// #include <pybind11/gil.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/pytypes.h>

// #include <string>  // IWYU pragma: keep

// IWYU pragma: no_include <gtest/gtest-message.h>
// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <gtest/gtest-test-part.h>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <tupleobject.h>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;
using namespace mrc::codable;
using namespace pybind11::literals;

PYMRC_TEST_CLASS(CodablePyobject);

TEST_F(TestCodablePyobject, PyObject)
{
    static_assert(is_codable_v<pymrc::PyHolder>, "pybind11::object should be codable.");
    static_assert(is_codable_v<py::object>, "pybind11::object should be codable.");
    static_assert(is_decodable_v<py::object>, "pybind11::object should be decodable.");
    static_assert(is_decodable_v<pymrc::PyHolder>, "pybind11::object should be decodable.");
    static_assert(!is_codable_v<PyObject>,
                  "No support for directly coding cpython objects -- "
                  "use pybind11::object or mrc::PyHolder");
    static_assert(!is_decodable_v<PyObject>,
                  "No support for directly coding cpython objects -- "
                  "use pybind11::object or mrc::PyHolder");
}

// todo(ryan/mdemoret) - reenable when python has a runtime object and a codable storage object can be acquired

// TEST_F(TestCodablePyobject, EncodedObjectSimple)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     py::object py_dict = py::dict("prop1"_a = py::none(),
//                                   "prop2"_a = 123,
//                                   "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                                   "prop4"_a = py::bool_(false),
//                                   "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);
//     encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     py::dict py_dict_deserialized = decode<py::object>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, EncodedHolderObjectSimple)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     pymrc::PyHolder py_dict = py::dict("prop1"_a = py::none(),
//                                        "prop2"_a = 123,
//                                        "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                                        "prop4"_a = py::bool_(false),
//                                        "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);
//     encode(py_dict, enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     pymrc::PyHolder py_dict_deserialized = decode<pymrc::PyHolder>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, EncodedObjectSharedMem)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     py::dict py_dict("prop1"_a = py::none(),
//                      "prop2"_a = 123,
//                      "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                      "prop4"_a = py::bool_(false),
//                      "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, true);
//     encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     py::dict py_dict_deserialized = decode<py::object>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, EncodedHolderObjectSharedMem)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     pymrc::PyHolder py_dict = py::dict("prop1"_a = py::none(),
//                                        "prop2"_a = 123,
//                                        "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                                        "prop4"_a = py::bool_(false),
//                                        "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, true);
//     encode(py_dict, enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     pymrc::PyHolder py_dict_deserialized = decode<pymrc::PyHolder>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, EncodedObjectSharedMemNoCopy)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     py::dict py_dict("prop1"_a = py::none(),
//                      "prop2"_a = 123,
//                      "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                      "prop4"_a = py::bool_(false),
//                      "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);
//     encode(py::cast<py::object>(py_dict), enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     py::dict py_dict_deserialized = decode<py::object>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, EncodedHolderObjectSharedMemNoCopy)
// {
//     py::gil_scoped_acquire gil;

//     py::module_ mod = py::module_::import("os");

//     pymrc::PyHolder py_dict = py::dict("prop1"_a = py::none(),
//                                        "prop2"_a = 123,
//                                        "prop3"_a = py::dict("subprop1"_a = 1, "subprop2"_a = "abc"),
//                                        "prop4"_a = py::bool_(false),
//                                        "func"_a  = py::getattr(mod, "getuid"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);
//     encode(py_dict, enc_obj, enc_ops);

//     EXPECT_EQ(enc_obj.object_count(), 1);
//     EXPECT_EQ(enc_obj.descriptor_count(), 1);

//     pymrc::PyHolder py_dict_deserialized = decode<pymrc::PyHolder>(enc_obj);

//     EXPECT_TRUE(!py_dict_deserialized.copy_obj().equal(py::dict()));
//     EXPECT_TRUE(py_dict_deserialized.equal(py_dict));
//     EXPECT_TRUE(py_dict_deserialized["prop3"].equal(py_dict["prop3"]));
// }

// TEST_F(TestCodablePyobject, BadUnpickleable)
// {
//     py::gil_scoped_acquire gil;
//     py::dict py_dict("mod(unpickleable)"_a = py::module_::import("sys"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);

//     EXPECT_THROW(encode(py::cast<py::object>(py_dict), enc_obj, enc_ops), py::error_already_set);
// }

// TEST_F(TestCodablePyobject, BadHolderUnpickleable)
// {
//     py::gil_scoped_acquire gil;
//     pymrc::PyHolder py_dict = py::dict("mod(unpickleable)"_a = py::module_::import("sys"));

//     EncodedObject enc_obj;
//     EncodingOptions enc_ops(true, false);

//     EXPECT_THROW(encode(py_dict, enc_obj, enc_ops), py::error_already_set);
// }
