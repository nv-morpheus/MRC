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

#include <ucp/api/ucp.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>

static constexpr ucp_tag_t ALL1_BITS = 0xFFFFFFFFFFFFFFFF;  // NOLINT
static constexpr ucp_tag_t ALL0_BITS = 0x0000000000000000;  // NOLINT

static constexpr ucp_tag_t TAG_MSG_MASK = 0xF000000000000000;  // leading 4 bits are 1111  // NOLINT
static constexpr ucp_tag_t TAG_RND_MSG  = 0x8000000000000000;  // leading 4 bits are 1000  // NOLINT
static constexpr ucp_tag_t TAG_EGR_MSG  = 0x4000000000000000;  // leading 4 bits are 0100  // NOLINT
static constexpr ucp_tag_t TAG_P2P_MSG  = 0x2000000000000000;  // leading 4 bits are 0010  // NOLINT
static constexpr ucp_tag_t TAG_UKN_MSG  = 0x1000000000000000;  // leading 4 bits are 0001  // NOLINT

static constexpr ucp_tag_t TAG_CTRL_MASK = 0xFFFF000000000000;  // 48-bits  // NOLINT
static constexpr ucp_tag_t TAG_USER_MASK = 0x0000FFFFFFFFFFFF;  // 48-bits  // NOLINT

static ucp_tag_t decode_tag_msg(const ucp_tag_t& tag)
{
    // valid = non-zero && power of two
    auto top_bits = tag & TAG_MSG_MASK;
    if ((top_bits & top_bits - 1) != 0)
    {
        throw std::runtime_error("invalid top_bits tag");
    }
    return top_bits;
}

static std::uint64_t decode_user_bits(const ucp_tag_t& tag)
{
    return tag & TAG_USER_MASK;
}
