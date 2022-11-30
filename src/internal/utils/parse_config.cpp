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

#include "./parse_config.hpp"

#include "./parse_ints.hpp"

#include <glog/logging.h>

#include <cstdint>  // for uint32_t
#include <cstdlib>  // for atoi
#include <iostream>
#include <memory>  // std::atoi(tokens[1].c_str()); is using allocator_traits
#include <stdexcept>
#include <utility>  // for move

namespace {

std::vector<std::string> split_string_on(std::string str, char delim)
{
    std::vector<std::string> tokens;
    std::istringstream f(str);
    std::string s;
    while (std::getline(f, s, delim))
    {
        tokens.push_back(s);
    }
    return tokens;
}
}  // namespace

namespace mrc {

ConfigurationMap parse_config(std::string config_str)
{
    ConfigurationMap config;

    bool left_wildcard = false;

    for (const auto& entry : split_string_on(config_str, ';'))
    {
        auto tokens = split_string_on(entry, ':');

        int concurrency = 1;
        std::vector<std::string> s;
        std::set<std::string> segments;
        std::vector<std::uint32_t> groups;

        switch (tokens.size())
        {
        case 3:
            if (tokens[2] != "*")
            {
                auto ints = parse_ints(tokens[2]);
                for (const auto& i : ints)
                {
                    CHECK_GE(i, 0);
                    groups.push_back(i);
                }
            }

        case 2:
            concurrency = std::atoi(tokens[1].c_str());
        case 1:
            // parse segments
            s = split_string_on(tokens[0], ',');
            segments.insert(s.begin(), s.end());
            break;

        default:
            throw std::runtime_error(
                "misformed user configuration string; should be of the form "
                "<segment_set>:<concurrency=1>:<group_set:*>;[repeated]");
        }

        config.push_back(std::make_tuple(std::move(segments), concurrency, std::move(groups)));
    }

    return config;
}

}  // namespace mrc
