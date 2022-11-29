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

#include "mrc/channel/status.hpp"
#include "mrc/channel/types.hpp"

namespace mrc::channel {

/**
 * @brief Interface for data flowing out of a channel via a fiber yielding read mechanism
 *
 * @tparam T
 */
template <typename T>
struct Egress
{
    virtual ~Egress() = default;

    virtual Status await_read(T&)                            = 0;
    virtual Status await_read_until(T&, const time_point_t&) = 0;
    virtual Status try_read(T&)                              = 0;
};

}  // namespace mrc::channel
