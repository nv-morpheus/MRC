/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/utils/macros.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>

namespace {
class A
{
  public:
    A(int val) : val(val) {}
    int val;
};

class B : public A
{
  public:
    B(int val) : A(val) {}
};
}  // namespace

TEST_CLASS(Macros);

TEST_F(TestMacros, MRC_PTR_CAST)
{
    // We can't test the fail case as that terminates,
    // in addition to that we run tests in CI against release not debug builds
    auto b_ptr = std::make_shared<B>(5);
    auto a_ptr = MRC_PTR_CAST(A, b_ptr);

    EXPECT_NE(a_ptr, nullptr);
    EXPECT_EQ(a_ptr->val, 5);
}
