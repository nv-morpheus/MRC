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

#include "mrc/core/addresses.hpp"

#include "mrc/types.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <ostream>
#include <string>

namespace mrc {

namespace {

template <typename S>
struct fnv_internal;

template <typename S>
struct fnv1a;

template <>
struct fnv_internal<std::uint32_t>
{
    constexpr static std::uint32_t DefaultOffsetBasis = 0x811C9DC5;
    constexpr static std::uint32_t Prime              = 0x01000193;
};

template <>
struct fnv1a<std::uint32_t> : public fnv_internal<std::uint32_t>
{
    constexpr static inline std::uint32_t hash(char const* const a_string, const std::uint32_t val = DefaultOffsetBasis)
    {
        return (a_string[0] == '\0') ? val : hash(&a_string[1], (val ^ std::uint32_t(a_string[0])) * Prime);
    }
};

std::uint16_t hash_16(const std::string& name)
{
    auto hash_u32          = fnv1a<std::uint32_t>::hash(name.c_str());
    std::uint16_t hash_u16 = (hash_u32 & 0x0000FFFF);
    return hash_u16;
}

}  // namespace

std::uint16_t segment_name_hash(const std::string& name)
{
    return hash_16(name);
}

std::uint16_t port_name_hash(const std::string& name)
{
    return hash_16(name);
}

SegmentAddress segment_address_encode(SegmentID seg_id, SegmentRank seg_rank)
{
    SegmentAddress i = seg_id;
    SegmentAddress r = seg_rank;
    return (i << 16 | r);
}

std::tuple<SegmentID, SegmentRank> segment_address_decode(const SegmentAddress& address)
{
    std::uint16_t id   = (address >> 16) & 0x0000FFFF;
    std::uint16_t rank = address & 0x0000FFFF;
    return std::make_tuple(id, rank);
}

std::string segment_address_string(const SegmentID& id, const SegmentRank& rank)
{
    std::stringstream ss;
    ss << "[segment id: " << id << "; rank: " << rank << "]";
    return ss.str();
}

std::string segment_address_string(const SegmentAddress& address)
{
    auto [id, rank] = segment_address_decode(address);
    return segment_address_string(id, rank);
}

PortAddress port_address_encode(const SegmentAddress& seg_addr, const PortID& port)
{
    std::uint64_t a = seg_addr;
    std::uint64_t p = port;
    return (a << 16 | p);
}

PortAddress port_address_encode(const SegmentID& seg_id, const SegmentRank& seg_rank, const PortID& port)
{
    // std::uint64_t i = seg_id, r = seg_rank, p = port;
    // return (i << 32 | r << 16 | p);
    return port_address_encode(segment_address_encode(seg_id, seg_rank), port);
}

std::tuple<SegmentID, SegmentRank, PortID> port_address_decode(const PortAddress& address)
{
    DCHECK_EQ(address & 0xFFFF000000000000, 0);  // ensure upper 16 bits are not set
    std::uint16_t id   = (address >> 32) & 0x000000000000FFFF;
    std::uint16_t rank = (address >> 16) & 0x000000000000FFFF;
    std::uint16_t port = address & 0x000000000000FFFF;
    return std::make_tuple(id, rank, port);
}

std::string port_address_string(const SegmentID& id, const SegmentRank& rank, const PortID& port)
{
    std::stringstream ss;
    ss << "[segment id: " << id << "; rank: " << rank << "; port: " << port << "]";
    return ss.str();
}

std::string port_address_string(const PortAddress& address)
{
    auto [id, rank, port] = port_address_decode(address);
    return port_address_string(id, rank, port);
}

}  // namespace mrc
