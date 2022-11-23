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

#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/operators/operator.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/type_traits.hpp"

#include <memory>
#include <vector>

namespace mrc::node {

template <typename T>
class Broadcast : public Operator<T>, public SourceProperties<T>
{
  public:
    Broadcast(bool deep_copy = false) : m_deep_copy(deep_copy) {}
    ~Broadcast() = default;

  protected:
    // Operator::on_next
    channel::Status on_next(T&& data) override
    {
        for (int i = 1; i < m_output_channels.size(); ++i)
        {
            if constexpr (is_shared_ptr<T>::value)
            {
                if (m_deep_copy)
                {
                    auto deep_copy = std::make_shared<typename T::element_type>(*data);
                    CHECK(m_output_channels[i]->await_write(std::move(deep_copy)) == channel::Status::success);
                    continue;
                }
            }

            T shallow_copy(data);
            CHECK(m_output_channels[i]->await_write(std::move(shallow_copy)) == channel::Status::success);
        }

        return m_output_channels[0]->await_write(std::move(data));
    }

    // Operator::on_complete
    void on_complete() final
    {
        VLOG(10) << "Closing broadcast with " << m_output_channels.size() << " downstream channels";
        m_output_channels.clear();
    }

    void complete_edge(std::shared_ptr<channel::IngressHandle> ingress) override
    {
        auto typed_ingress = std::dynamic_pointer_cast<channel::Ingress<T>>(ingress);

        CHECK(typed_ingress) << "Invalid ingress type passed to broadcast";

        m_output_channels.push_back(typed_ingress);
    }

    std::vector<std::shared_ptr<channel::Ingress<T>>> m_output_channels;
    bool m_deep_copy;
};

}  // namespace mrc::node
