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

#include "./parse_ints.hpp"

#include "mrc/utils/string_utils.hpp"  // for split_string_to_vector

#include <glog/logging.h>

#include <iostream>

namespace {

int convert_string2_int(const std::string& str)
{
    int x;
    std::stringstream ss(str);
    CHECK(ss >> x) << "Error converting " << str << " to integer";
    return x;
}

}  // namespace

namespace mrc {

std::vector<int> parse_ints(const std::string& data)
{
    std::vector<int> result;
    std::vector<std::string> tokens = split_string_to_vector(data, ",");
    for (auto& token : tokens)
    {
        std::vector<std::string> range = split_string_to_vector(token, "-");
        if (range.size() == 1)
        {
            result.push_back(convert_string2_int(range[0]));
        }
        else if (range.size() == 2)
        {
            int start = convert_string2_int(range[0]);
            int stop  = convert_string2_int(range[1]);
            for (int i = start; i <= stop; i++)
            {
                result.push_back(i);
            }
        }
        else
        {
            LOG(FATAL) << "Error parsing token " << token;
        }
    }
    return result;
}

}  // namespace mrc
