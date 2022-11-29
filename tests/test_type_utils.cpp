/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/utils/type_utils.hpp"

#include <gtest/gtest.h>  // for EXPECT_EQ

#include <vector>

TEST_CLASS(TypeUtils);

TEST_F(TestTypeUtils, DataTypeCopy)
{
    mrc::DataType d1(mrc::TypeId::INT32);
    mrc::DataType d2(mrc::TypeId::FLOAT32);

    std::vector<mrc::DataType> type_list;
    type_list.push_back(d1);
    type_list.push_back(d2);
    type_list.emplace_back(mrc::TypeId::INT32);
    type_list.emplace_back(mrc::TypeId::FLOAT32);

    EXPECT_EQ(type_list[0], d1);
    EXPECT_EQ(type_list[1], d2);
    EXPECT_EQ(type_list[2], d1);
    EXPECT_EQ(type_list[3], d2);

    mrc::DataType d3 = d1;
    mrc::DataType d4 = d2;

    EXPECT_EQ(d3, d1);
    EXPECT_EQ(d3.type_id(), d1.type_id());

    EXPECT_EQ(d4, d2);
    EXPECT_EQ(d4.type_id(), d2.type_id());

    mrc::DataType d5{d1};
    mrc::DataType d6{d2};
}
