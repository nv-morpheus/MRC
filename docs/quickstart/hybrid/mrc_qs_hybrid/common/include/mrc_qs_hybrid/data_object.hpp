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

#include <mrc/utils/string_utils.hpp>

#include <string>

namespace mrc::quickstart::hybrid::common {

struct DataObject
{
    DataObject(std::string n = "", int v = 0) : name(std::move(n)), value(v) {}

    std::string to_string() const
    {
        return MRC_CONCAT_STR("{Name: '" << this->name << "', Value: " << this->value << "}");
    }

    std::string name;
    int value{0};
};
}  // namespace mrc::quickstart::hybrid::common
