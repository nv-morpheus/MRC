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

#include "mrc/channel/channel.hpp"

#include "mrc/constants.hpp"

#include <cstddef>
#include <stdexcept>

namespace mrc::channel {

static std::size_t s_default_channel_size = MRC_DEFAULT_BUFFERED_CHANNEL_SIZE;

std::size_t default_channel_size()
{
    return s_default_channel_size;
}

void set_default_channel_size(std::size_t default_size)
{
    if (default_size < 2 || ((default_size & (default_size - 1)) != 0))
    {
        throw std::invalid_argument("default_channel_size must be greater than 1 and a power of 2.");
    }
    s_default_channel_size = default_size;
}

ChannelBase::~ChannelBase() = default;

}  // namespace mrc::channel
