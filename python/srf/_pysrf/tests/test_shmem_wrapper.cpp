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

#include "pysrf/module_wrappers/shared_memory.hpp"

#include <gtest/gtest.h>
#include <pybind11/buffer_info.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstring>
#include <stdexcept>
#include <string>

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"

namespace py = pybind11;
using namespace std::string_literals;
using namespace pybind11::literals;

PYSRF_TEST_CLASS(ShmemWrapper);

TEST_F(TestShmemWrapper, Construct)
{
    auto shmem = pysrf::PythonSharedMemoryInterface();
}

TEST_F(TestShmemWrapper, Allocate)
{
    auto shmem = pysrf::PythonSharedMemoryInterface();
    shmem.allocate(128);

    py::bytes bytes = shmem.get_bytes();
    py::buffer_info buf_info(py::buffer(bytes).request());
    ASSERT_TRUE(buf_info.size == 128);
    ASSERT_TRUE(py::int_(buf_info.size).equal(shmem.size_bytes()));

    shmem.free();
}

TEST_F(TestShmemWrapper, Attach)
{
    auto shmem1 = pysrf::PythonSharedMemoryInterface();
    auto shmem2 = pysrf::PythonSharedMemoryInterface();

    shmem1.allocate(128);

    py::bytes bytes1 = shmem1.get_bytes();
    py::buffer_info buf_info1(py::buffer(bytes1).request());

    shmem2.attach(shmem1.block_id());
    py::bytes bytes2 = shmem1.get_bytes();
    py::buffer_info buf_info2(py::buffer(bytes2).request());

    ASSERT_TRUE(buf_info2.size == 128);
    ASSERT_TRUE(std::memcmp(buf_info1.ptr, buf_info2.ptr, 128) == 0);

    char byteset[128];             // NOLINT
    char bytesubset[7]{"abc123"};  // NOLINT
    std::memcpy(byteset, bytesubset, 6);
    py::bytes py_byteset(byteset, 128);

    shmem1.set(py_byteset);
    bytes1    = shmem1.get_bytes();
    buf_info1 = py::buffer_info(py::buffer(bytes1).request());
    ASSERT_TRUE(std::memcmp(buf_info1.ptr, byteset, 6) == 0);

    bytes2    = shmem2.get_bytes();
    buf_info2 = py::buffer_info(py::buffer(bytes2).request());
    ASSERT_TRUE(std::memcmp(buf_info2.ptr, byteset, 6) == 0);
    ASSERT_TRUE(std::memcmp(buf_info1.ptr, buf_info2.ptr, 128) == 0);

    shmem1.close();
    shmem2.free();
}

TEST_F(TestShmemWrapper, CloseUnlink)
{
    auto shmem = pysrf::PythonSharedMemoryInterface();
    shmem.allocate(128);

    auto block_id      = shmem.block_id();
    py::bytes py_bytes = shmem.get_bytes();
    py::buffer_info buf_info(py::buffer(py_bytes).request());

    shmem.close();
    EXPECT_THROW(shmem.get_bytes(), std::runtime_error);

    shmem.attach(block_id);
    ASSERT_TRUE(py::int_(buf_info.size).equal(shmem.size_bytes()));

    py::bytes py_bytes2 = shmem.get_bytes();
    py::buffer_info buf_info2(py::buffer(py_bytes2).request());
    ASSERT_TRUE(std::memcmp(buf_info.ptr, buf_info2.ptr, 128) == 0);

    shmem.unlink();
    EXPECT_THROW(shmem.attach(block_id), pybind11::error_already_set);
}
