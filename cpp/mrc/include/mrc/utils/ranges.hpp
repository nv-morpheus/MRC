/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mrc {

template <typename T>
std::vector<std::pair<T, T>> find_ranges(const std::vector<T>& values)
{
    static_assert(std::is_integral<T>::value, "only integral types allowed");

    auto copy = values;
    std::sort(copy.begin(), copy.end());

    std::vector<std::pair<T, T>> ranges;

    auto it  = copy.cbegin();
    auto end = copy.cend();

    while (it != end)
    {
        auto low  = *it;
        auto high = *it;
        for (T i = 0; it != end && low + i == *it; it++, i++)
        {
            high = *it;
        }
        ranges.push_back(std::make_pair(low, high));
    }

    return ranges;
}

template <typename T>
std::string print_ranges(const std::vector<std::pair<T, T>>& ranges)
{
    return std::accumulate(std::begin(ranges), std::end(ranges), std::string(), [](std::string r, std::pair<T, T> p) {
        if (p.first == p.second)
        {
            return r + (r.empty() ? "" : ",") + std::to_string(p.first);
        }

        return r + (r.empty() ? "" : ",") + std::to_string(p.first) + "-" + std::to_string(p.second);
    });
}

// Returns a tuple of [added, removed]
template <typename T>
std::pair<std::set<T>, std::set<T>> compare_difference(const std::set<T>& cur_set, const std::set<T>& new_set)
{
    std::set<T> remove;
    std::set<T> create;

    // set difference to determine which channels to remove
    std::set_difference(cur_set.begin(),
                        cur_set.end(),
                        new_set.begin(),
                        new_set.end(),
                        std::inserter(remove, remove.end()));

    // set difference to determine which channels to add
    std::set_difference(new_set.begin(),
                        new_set.end(),
                        cur_set.begin(),
                        cur_set.end(),
                        std::inserter(create, create.end()));

    return std::make_pair(std::move(create), std::move(remove));
}

// Returns a tuple of [added, removed]
template <typename T>
std::pair<std::set<T>, std::set<T>> compare_difference(const std::vector<T>& curr, const std::vector<T>& next)
{
    // Convert the vectors to sets
    std::set<T> curr_set(curr.begin(), curr.end());
    std::set<T> next_set(next.begin(), next.end());

    return compare_difference(curr_set, next_set);
}

// Returns a tuple of [duplicates, unique]
template <typename T>
std::pair<std::set<T>, std::set<T>> compare_intersecton(const std::set<T>& curr, const std::set<T>& next)
{
    std::set<T> duplicates;
    std::set<T> unique;

    std::set_intersection(curr.begin(),
                          curr.end(),
                          next.begin(),
                          next.end(),
                          std::inserter(duplicates, duplicates.end()));

    std::set_symmetric_difference(curr.begin(),
                                  curr.end(),
                                  next.begin(),
                                  next.end(),
                                  std::inserter(unique, unique.end()));

    return std::make_pair(std::move(duplicates), std::move(unique));
}

// Returns a tuple of [duplicates, unique]
template <typename T>
std::pair<std::set<T>, std::set<T>> compare_intersecton(const std::vector<T>& curr, const std::vector<T>& next)
{
    // Convert the vectors to sets
    std::set<T> curr_set(curr.begin(), curr.end());
    std::set<T> next_set(next.begin(), next.end());

    return compare_intersecton(curr_set, next_set);
}

}  // namespace mrc
