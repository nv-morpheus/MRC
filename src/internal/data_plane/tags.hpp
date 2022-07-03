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
#include <tuple>

static constexpr ucp_tag_t ALL1_BITS = 0xFFFFFFFFFFFFFFFF;  // NOLINT
static constexpr ucp_tag_t ALL0_BITS = 0x0000000000000000;  // NOLINT

static constexpr ucp_tag_t MEM_TYPE_MASK = 0x0100000000000000;  // leading 8 bits are 0000 0001  // NOLINT
static constexpr ucp_tag_t DEVICE_TAG    = 0x0100000000000000;  // leading 8 bits are 0000 0001  // NOLINT
static constexpr ucp_tag_t HOST_TAG      = 0x0000000000000000;  // leading 8 bits are 0000 0000  // NOLINT

static constexpr ucp_tag_t MSG_TYPE_MASK  = 0xF000000000000000;  // leading 4 bits are 1111  // NOLINT
static constexpr ucp_tag_t INGRESS_TAG    = 0x8000000000000000;  // leading 4 bits are 1000  // NOLINT
static constexpr ucp_tag_t DESCRIPTOR_TAG = 0x4000000000000000;  // leading 4 bits are 0100  // NOLINT
static constexpr ucp_tag_t FUTURE_TAG     = 0x2000000000000000;  // leading 4 bits are 0010  // NOLINT
static constexpr ucp_tag_t P2P_TAG        = 0x1000000000000000;  // leading 4 bits are 0010  // NOLINT

static constexpr ucp_tag_t TAG_CTRL_MASK = 0xFFFF000000000000;  // 48-bits  // NOLINT
static constexpr ucp_tag_t TAG_USER_MASK = 0x0000FFFFFFFFFFFF;  // 48-bits  // NOLINT
static constexpr ucp_tag_t USR_TYPE_MASK = 0x0000FFFFFFFFFFFF;  // 48-bits  // NOLINT

static ucp_tag_t tag_decode_msg_type(const ucp_tag_t& tag)
{
    auto top_bits = tag & MSG_TYPE_MASK;
    // valid = non-zero && power of two
    if ((top_bits & top_bits - 1) != 0)
    {
        return 0;
    }
    return top_bits;
}

static ucp_tag_t tag_decode_mem_type(const ucp_tag_t& tag)
{
    return tag & MEM_TYPE_MASK;
}

static std::uint64_t tag_decode_user_tag(const ucp_tag_t& tag)
{
    return tag & USR_TYPE_MASK;
}

static std::tuple<ucp_tag_t, ucp_tag_t, std::uint8_t, std::uint32_t> tag_decode(const ucp_tag_t& tag)
{
    auto msg_tag = tag_decode_msg_type(tag);
    auto mem_tag = tag_decode_mem_type(tag);
    return std::make_tuple(msg_tag, mem_tag, 0, tag_decode_user_tag(tag));
}

// proposal
// use high 4 bits 0x1, 0x2, 0x4, 0x8

// 0x8 = node id send/recv
// 0x4 = obj id dec/[inc]
// 0x2 = future / promise
// 0x1 = unused

// 0x08 = unused
// 0x04 = unused
// 0x02 = unused
// 0x01 = memory_type; 0=host; 1=cuda

// match any bit pattern

// mask low bits - 60 out and check for power of two (v & v - 1) == 0
// ensure only 1 of the first 3-6 high bits is set, currently only using 3
