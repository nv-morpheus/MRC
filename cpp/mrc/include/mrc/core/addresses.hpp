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

#include "mrc/types.hpp"

#include <cstdint>
#include <string>
#include <tuple>

namespace mrc {

/**
 * @brief Hash of segment name
 *
 * @param name
 * @return std::uint16_t
 */
std::uint16_t segment_name_hash(const std::string& name);

/**
 * @brief Hash of port name
 *
 * @param name
 * @return std::uint16_t
 */
std::uint16_t port_name_hash(const std::string& name);

/**
 * @brief Encodes a SegmentAddress from a SegmentID and SegmentRank
 * @param [in] id
 * @param [in] rank
 * @returns SegmentAddress
 **/
extern SegmentAddress segment_address_encode(SegmentID id, SegmentRank rank);

/**
 * @brief Decodes a SegmentAddress
 * @param [in] segment_address
 * @returns std::tuple<SegmentID, SegmentRank>
 **/
extern std::tuple<SegmentID, SegmentRank> segment_address_decode(const SegmentAddress& address);

/**
 * @brief returns a string describing a SegementPortAddress' components
 **/
extern std::string segment_address_string(const SegmentID& id, const SegmentRank& rank);

/**
 * @brief returns a string describing a PortAddress
 **/
extern std::string segment_address_string(const SegmentAddress& address);

/**
 * @brief Creates an Encoded PortAddress
 * @param [in]  id     segment id
 * @param [in]  rank   segment rank
 * @param [in]  port   port id
 * @returns     addr   encoded address
 */
extern PortAddress port_address_encode(const SegmentID& seg_id, const SegmentRank& seg_rank, const PortID& port);

/**
 * @brief Encodes a PortAddress from a SegmentAddress and a PortID
 *
 * @param seg_addr
 * @param port
 * @return PortAddress
 */
extern PortAddress port_address_encode(const SegmentAddress& seg_addr, const PortID& port);

/**
 * @brief Decodes a PortAddress
 * @param [in]  addr   encoded address
 * @returns     tuple  {id, rank, port}
 */
extern std::tuple<SegmentID, SegmentRank, PortID> port_address_decode(const PortAddress& address);

/**
 * @brief returns a string describing a SegementPortAddress' components
 **/
extern std::string port_address_string(const SegmentID& id, const SegmentRank& rank, const PortID& port);

/**
 * @brief returns a string describing a PortAddress
 **/
extern std::string port_address_string(const PortAddress& address);

}  // namespace mrc
