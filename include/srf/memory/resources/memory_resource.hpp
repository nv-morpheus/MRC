/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "srf/memory/memory_kind.hpp"

#include <cuda/memory_resource>

#include <string>
#include <vector>

namespace srf::memory {

template <typename MemoryKind>
class memory_resource : public ::cuda::memory_resource<MemoryKind>
{
  public:
    memory_resource(std::string tag)
    {
        add_tag(tag);
    }

    const std::string& tag() const
    {
        return m_tags[m_tags.size() - 1];
    }

    const std::vector<std::string>& tags() const
    {
        return m_tags;
    }

    memory_kind_type kind() const
    {
        return do_kind();
    }

  protected:
    void add_tag(const std::string& tag)
    {
        m_tags.insert(m_tags.begin(), tag);
    }

    void add_tags(const std::vector<std::string>& tags)
    {
        m_tags.insert(m_tags.begin(), tags.begin(), tags.end());
    }

  private:
    virtual memory_kind_type do_kind() const = 0;

    std::vector<std::string> m_tags;
};

}  // namespace srf::memory
