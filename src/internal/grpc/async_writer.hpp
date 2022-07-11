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

#include "srf/node/generic_node.hpp"

#include <boost/fiber/all.hpp>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/async_stream.h>

#pragma once

namespace srf::internal::rpc {

template <typename T>
class Writer final : public srf::node::GenericNode<T, bool>
{
  public:
    Writer(std::shared_ptr<grpc::internal::AsyncWriterInterface<T>> writer) : m_writer(std::move(writer))
    {
        CHECK(m_writer);
    }

  private:
    void on_data(T&& data, rxcpp::subscriber<bool>& subscriber) final
    {
        if (m_able_to_write)  // todo(cpp20) [[likely]]
        {
            boost::fibers::promise<bool> promise;
            m_writer->Write(data, &promise);
            auto ok = promise.get_future().get();
            if (!ok)  // todo(cpp20) [[unlikely]]
            {
                m_able_to_write = false;
            }
        }
    }

    void on_completed(rxcpp::subscriber<bool>& subscriber) final
    {
        subscriber.on_next(m_able_to_write);
    }

    void on_stop(const rxcpp::subscription& subscription) const final {}

    bool m_able_to_write{true};
    const std::shared_ptr<grpc::internal::AsyncWriterInterface<T>> m_writer;
};

}  // namespace srf::internal::rpc
