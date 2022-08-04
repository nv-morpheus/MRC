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

#include "internal/control_plane/server/tagged_service.hpp"

#include <glog/logging.h>

namespace srf::internal::control_plane::server {

Tagged::~Tagged() = default;

Tagged::tag_t Tagged::upper_bound() const
{
    return (m_tag + UINT16_MAX);
}
Tagged::tag_t Tagged::lower_bound() const
{
    return m_tag;
}
bool Tagged::valid_tag(const tag_t& tag) const
{
    static constexpr std::uint64_t Mask = 0x0000FFFFFFFF0000;
    return ((tag & Mask) == m_tag);
}
Tagged::tag_t Tagged::next_tag()
{
    if (m_uid++ < UINT16_MAX)
    {
        return (m_tag + m_uid);
    }
    throw std::overflow_error(
        SRF_CONCAT_STR("limit of uniquely Tagged objects with tag " << m_tag << " reached; fatal error"));
}
Tagged::tag_t Tagged::next()
{
    constexpr std::uint32_t MaxVal = 0x0FFFFFFF;
    static std::uint32_t next_tag  = 0;
    if (++next_tag < MaxVal)
    {
        std::uint64_t tag = next_tag;
        return (tag << 16);
    }
    throw std::overflow_error("limit of Taggable objects reached; fatal error");
}

TaggedService::~TaggedService()
{
    if (!m_instance_tags.empty())
    {
        LOG(FATAL) << "TaggedService destructor called before all tagged instances were released";
    }
}

void TaggedService::drop_all()
{
    if (!m_instance_tags.empty())
    {
        DVLOG(10) << "TaggedService: dropping remaining tags";
        for (auto i = m_instance_tags.begin(); i != m_instance_tags.end();)
        {
            i = drop_tag(i);
        }
    }
}

void TaggedService::drop_instance(std::shared_ptr<ClientInstance> instance)
{
    drop_instance(instance->get_id());
}
void TaggedService::drop_instance(ClientInstance::instance_id_t instance_id)
{
    DVLOG(10) << "dropping all tags for instance_id: " << instance_id;
    auto tags = m_instance_tags.equal_range(instance_id);
    for (auto i = tags.first; i != tags.second;)
    {
        i = drop_tag(i);
    }
}
void TaggedService::drop_tag(tag_t tag)
{
    for (auto i = m_instance_tags.begin(); i != m_instance_tags.end(); i++)
    {
        if (i->second == tag)
        {
            drop_tag(i);
            return;
        }
    }
    throw std::invalid_argument(SRF_CONCAT_STR("tag " << tag << " not registered"));
}
Tagged::tag_t TaggedService::register_instance_id(ClientInstance::instance_id_t instance_id)
{
    auto tag = next_tag();
    m_instance_tags.emplace(instance_id, tag);
    return tag;
}
decltype(TaggedService::m_instance_tags)::iterator TaggedService::drop_tag(decltype(m_instance_tags)::iterator it)
{
    DVLOG(10) << "dropping tag: " << it->second;
    do_drop_tag(it->second);
    return m_instance_tags.erase(it);
}
std::size_t TaggedService::tag_count_for_instance_id(ClientInstance::instance_id_t instance_id) const
{
    auto tags         = m_instance_tags.equal_range(instance_id);
    std::size_t count = 0;
    for (auto i = tags.first; i != tags.second; ++i)
    {
        count++;
    }
    return count;
}
std::size_t TaggedService::tag_count() const
{
    return m_instance_tags.size();
}

void TaggedService::issue_update()
{
    do_issue_update();
}

}  // namespace srf::internal::control_plane::server
