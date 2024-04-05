/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/utils/string_utils.hpp"

#include <boost/algorithm/string.hpp>  // for split

namespace mrc {
std::vector<std::string> split_string_to_array(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> results;
    boost::split(results, str, boost::is_any_of(delimiter));
    return results;
}

}  // namespace mrc
