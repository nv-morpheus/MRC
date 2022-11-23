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

#include "mrc/utils/bytes_to_string.hpp"

#include <glog/logging.h>

// cmath for std::log & std::pow
#include <cmath>    // IWYU pragma: keep
#include <cstddef>  // for size_t
#include <cstdint>  // for uint64_t
#include <cstdio>   // for sprintf
#include <map>
#include <ostream>  // for logging
#include <regex>
// IWYU thinks regex_search needs vector appears to be a known issue:
// https://github.com/include-what-you-use/include-what-you-use/issues/902
// IWYU pragma: no_include <vector>

namespace mrc {

std::string bytes_to_string(size_t bytes)
{
    char buffer[50];  // NOLINT
    int unit              = 1024;
    const char prefixes[] = "KMGTPE";  // NOLINT
    if (bytes < unit)
    {
        sprintf(buffer, "%ld B", bytes);
        return std::string(buffer);
    }
    int exp = (int)(std::log(bytes) / std::log(unit));
    sprintf(buffer, "%.1f %ciB", bytes / std::pow(unit, exp), prefixes[exp - 1]);
    return std::string(buffer);
}

std::uint64_t string_to_bytes(std::string str)
{
    // https://regex101.com/r/UVm5wT/1
    std::smatch m;
    std::regex r("(\\d+[.\\d+]*)([KMGTkmgt]*)([i]*)[bB]");
    std::map<char, int> prefix = {
        {'k', 1},
        {'m', 2},
        {'g', 3},
        {'t', 4},
        {'K', 1},
        {'M', 2},
        {'G', 3},
        {'T', 4},
    };

    if (!std::regex_search(str, m, r))
    {
        LOG(FATAL) << "Unable to convert \"" << str << "\" to bytes. "
                   << "Expected format: 10b, 1024B, 1KiB, 10MB, 2.4gb, etc.";
    }

    const std::uint64_t base = m.empty() || (m.size() > 3 && m[3] == "") ? 1000 : 1024;
    auto exponent            = prefix[m[2].str()[0]];
    auto scalar              = std::stod(m[1]);
    return (std::uint64_t)(scalar * std::pow(base, exponent));
}

}  // namespace mrc
