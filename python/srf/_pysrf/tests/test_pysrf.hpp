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

#pragma once

#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"  // IWYU pragma: export
#include "srf/memory/resources/host/pinned_memory_resource.hpp"  // IWYU pragma: export

#include <gtest/gtest.h>
#include <pybind11/embed.h>

using namespace mrc;
// Essentially the same macro from test_srf.hpp but with an embedded python interpreter.
// The attribute visibility bit avoids a compiler warning about the test class being
// declared with greater visibility than the interpreter attribute
#define PYSRF_TEST_CLASS(name)                                                       \
    class __attribute__((visibility("default"))) Test##name : public ::testing::Test \
    {                                                                                \
        py::scoped_interpreter m_interpreter;                                        \
    }
