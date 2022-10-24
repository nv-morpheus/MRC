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

#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/constants.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_channel.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/utils/type_utils.hpp"

#include <memory>

namespace srf::node {

/**
 * @brief Extends SourceProperties to hold a channel ingress which is the writing interface to an edge.
 *
 * @tparam T
 */
template <typename T>
class SourceChannel : public virtual SourceProperties<T>
{
  public:
    ~SourceChannel() override = default;

    void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
    {
        EdgeChannel<T> edge_channel(std::move(channel));

        this->do_set_channel(edge_channel);
    }

  protected:
    SourceChannel() = default;

    void do_set_channel(EdgeChannel<T>& edge_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_readable+edge
        auto channel_reader = edge_channel.get_reader();
        auto channel_writer = edge_channel.get_writer();

        channel_writer->add_connector(EdgeLifetime<T>([this, channel_reader]() {
            // On connection, save the reader so we can use the channel without it being deleted
            this->m_set_edge = channel_reader;
        }));

        SourceProperties<T>::init_edge(channel_writer);
    }

    //     inline channel::Status await_write(T&& data) final
    //     {
    //         if (m_ingress)
    //         {
    //             return m_ingress->await_write(std::move(data));
    //         }

    //         return no_channel(std::move(data));
    //     }

    //     bool has_channel() const
    //     {
    //         return bool(m_ingress);
    //     }

    //     void release_channel()
    //     {
    //         m_ingress.reset();
    //     }

    //   private:
    //     virtual channel::Status no_channel(T&& data)
    //     {
    //         LOG(ERROR) << "SourceChannel has either not been connected or the channel has been released";
    //         throw exceptions::SrfRuntimeError(
    //             "SourceChannel has either not been connected or the channel has been released");
    //         return channel::Status::error;
    //     }

    //     void complete_edge(std::shared_ptr<channel::IngressHandle> untyped_ingress) override
    //     {
    //         CHECK(untyped_ingress);
    //         if (m_ingress != nullptr)
    //         {
    //             // todo(ryan) - we could specialize this exception, then if we catch it in
    //             segment::Builder::make_edge, we
    //             // could enhance the error description and rethrow the same exception
    //             throw exceptions::SrfRuntimeError(
    //                 "multiple edges to a source detected; use an operator to select proper behavior");
    //         }
    //         m_ingress = std::dynamic_pointer_cast<channel::Ingress<T>>(untyped_ingress);
    //         CHECK(m_ingress);
    //     }

    //     std::shared_ptr<channel::Ingress<T>> m_ingress;
};

// template <typename T>
// class SourceChannelWriteable : public SourceChannel<T>
// {
//   public:
//     using SourceChannel<T>::await_write;

//   private:
//     channel::Status no_channel(T&& data) final
//     {
//         // unsubscribed - into the ether?
//         return channel::Status::success;
//     }
// };

}  // namespace srf::node
