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

#include "internal/control_plane/server/tagged_issuer.hpp"

#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>

#include <ostream>
#include <stdexcept>
#include <utility>

namespace mrc::internal::control_plane::server {

Tagged::~Tagged() = default;

TagID Tagged::upper_bound() const
{
    return (m_tag + UINT16_MAX);
}
TagID Tagged::lower_bound() const
{
    return m_tag;
}
bool Tagged::is_valid_tag(const TagID& tag) const
{
    static constexpr std::uint64_t Mask = 0x0000FFFFFFFF0000;
    return ((tag & Mask) == m_tag);
}
bool Tagged::is_issued_tag(const TagID& tag) const
{
    return (tag > lower_bound() && tag <= (m_tag + m_uid));
}
TagID Tagged::next_tag()
{
    if (m_uid++ < UINT16_MAX)
    {
        return (m_tag + m_uid);
    }
    throw std::overflow_error(
        MRC_CONCAT_STR("limit of uniquely Tagged objects with tag " << m_tag << " reached; fatal error"));
}
TagID Tagged::next()
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

TaggedIssuer::~TaggedIssuer()
{
    if (!m_instance_tags.empty())
    {
        LOG(FATAL) << "TaggedIssuer destructor called before all tagged instances were released";
    }
}

void TaggedIssuer::drop_all()
{
    if (!m_instance_tags.empty())
    {
        DVLOG(10) << "TaggedIssuer: dropping remaining tags";
        for (auto i = m_instance_tags.begin(); i != m_instance_tags.end();)
        {
            i = drop_tag(i);
        }
    }
}

void TaggedIssuer::drop_instance(std::shared_ptr<ClientInstance> instance)
{
    drop_instance(instance->get_id());
}
void TaggedIssuer::drop_instance(ClientInstance::instance_id_t instance_id)
{
    DVLOG(10) << "dropping all tags for instance_id: " << instance_id;
    auto tags = m_instance_tags.equal_range(instance_id);
    for (auto i = tags.first; i != tags.second;)
    {
        i = drop_tag(i);
    }
}
void TaggedIssuer::drop_tag(TagID tag)
{
    for (auto i = m_instance_tags.begin(); i != m_instance_tags.end(); i++)
    {
        if (i->second == tag)
        {
            drop_tag(i);
            return;
        }
    }
    throw std::invalid_argument(MRC_CONCAT_STR("tag " << tag << " not registered"));
}
TagID TaggedIssuer::register_instance_id(ClientInstance::instance_id_t instance_id)
{
    auto tag = next_tag();
    m_instance_tags.emplace(instance_id, tag);
    return tag;
}
decltype(TaggedIssuer::m_instance_tags)::iterator TaggedIssuer::drop_tag(decltype(m_instance_tags)::iterator it)
{
    DVLOG(10) << "dropping tag: " << it->second;
    do_drop_tag(it->second);
    return m_instance_tags.erase(it);
}
std::size_t TaggedIssuer::tag_count_for_instance_id(ClientInstance::instance_id_t instance_id) const
{
    return m_instance_tags.count(instance_id);
}
std::size_t TaggedIssuer::tag_count() const
{
    return m_instance_tags.size();
}

void TaggedIssuer::issue_update()
{
    do_issue_update();
}

}  // namespace mrc::internal::control_plane::server
