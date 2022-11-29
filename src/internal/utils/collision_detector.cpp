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

#include "internal/utils/collision_detector.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/exceptions/runtime_error.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <ostream>
#include <utility>

namespace mrc::internal::utils {

std::uint16_t CollisionDetector::register_name(const std::string& name)
{
    auto hash   = port_name_hash(name);
    auto search = m_hashes.find(hash);
    if (search == m_hashes.end())
    {
        // never used
        m_hashes[hash] = name;
    }
    else
    {
        // test for collision
        if (search->second != name)
        {
            LOG(ERROR) << "naming hash collision detected between " << search->second << " and " << name << ".\n"
                       << "Interally, 16-bits are used to hash names. Make a small modification to"
                       << "one of these names, e.g. adding an underscore or a dash";
            throw exceptions::MrcRuntimeError("name registration collision");
        }
    }
    return hash;
}

std::uint16_t CollisionDetector::lookup_name(const std::string& name) const
{
    auto hash   = port_name_hash(name);
    auto search = m_hashes.find(hash);
    if (search == m_hashes.end())
    {
        throw exceptions::MrcRuntimeError("name not registered");
    }

    // test for collision
    if (search->second != name)
    {
        LOG(ERROR) << "naming hash collision detected between " << search->second << " and " << name << ".\n"
                   << "Interally, 16-bits are used to hash names. Make a small modification to"
                   << "one of these names, e.g. adding an underscore or a dash";
        throw exceptions::MrcRuntimeError("name registration collision");
    }

    return hash;
}

const std::string& CollisionDetector::name(const std::uint16_t& hash) const
{
    auto search = m_hashes.find(hash);
    if (search == m_hashes.end())
    {
        throw exceptions::MrcRuntimeError("hash not registered");
    }
    return search->second;
}

}  // namespace mrc::internal::utils
