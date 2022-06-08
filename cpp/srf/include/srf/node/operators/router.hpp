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

#include <srf/exceptions/runtime_error.hpp>
#include <srf/node/forward.hpp>
#include <srf/node/operators/operator.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_channel.hpp>
#include <srf/node/source_properties.hpp>

#include <map>
#include <memory>

namespace srf::node {

template <typename KeyT, typename T>
class RouterBase
{
    std::map<KeyT, SourceChannelWriteable<T>> m_sources;

  protected:
    inline SourceChannelWriteable<T>& channel_for_key(const KeyT& key)
    {
        auto search = m_sources.find(key);
        if (search == m_sources.end())
        {
            throw exceptions::SrfRuntimeError("unable to find edge for key");
        }
        return search->second;
    }

    void release_sources()
    {
        m_sources.clear();
    }

  public:
    SourceChannel<T>& source(KeyT key)
    {
        return m_sources[key];
    }

    bool has_edge(KeyT key) const
    {
        auto search = m_sources.find(key);
        return (search != m_sources.end());
    }

    void drop_edge(KeyT key)
    {
        auto search = m_sources.find(key);
        if (search != m_sources.end())
        {
            m_sources.erase(search);
        }
    }
};

template <typename KeyT, typename T>
class Router : public Operator<std::pair<KeyT, T>>, public RouterBase<KeyT, T>
{
    // Operator::on_next
    inline channel::Status on_next(std::pair<KeyT, T>&& tagged_data) final
    {
        return this->channel_for_key(tagged_data.first).await_write(std::move(tagged_data.second));
    }

    // Operator::on_complete
    void on_complete() final
    {
        this->release_sources();
    }
};

}  // namespace srf::node
