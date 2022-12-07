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

#include <cstdint>
#include <map>
#include <string>

namespace mrc::internal::utils {

/**
 * @brief Creates 16-bit hash for a given string.
 *
 * The returned hash is 16-bit; however, the computed hash is 32-bit. This class holds the
 * string and the hash to help detect possible collisions
 **/
class CollisionDetector
{
  public:
    /**
     * @brief registers a port name to check for name collisions in the 16-bit hash-space
     * @param [in] name   port name to register
     * @returns    port  uint16_t port id associated with name
     * @throws     MrcRuntimeError in the event of a name collision
     **/
    std::uint16_t register_name(const std::string& name);

    /**
     * @brief lookup name in registry to determine if it registered or not
     * @param [in] name   port name to check registration
     * @returns    port  uint16_t port id associated with name
     * @throws     MrcRuntimeError in the event of name is not registred
     **/
    std::uint16_t lookup_name(const std::string& name) const;

    /**
     * @brief lookup name associated with hash
     * @param   [in] hash
     * @returns      name
     * @throws       MrcRuntimeError if hash is not registered
     **/
    const std::string& name(const std::uint16_t& hash) const;

  private:
    std::map<std::uint16_t, std::string> m_hashes;
};

}  // namespace mrc::internal::utils
