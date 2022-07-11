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

#include "srf/node/generic_source.hpp"

#include <boost/fiber/all.hpp>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/async_stream.h>

#pragma once

namespace srf::internal::rpc {

template <typename T>
class Reader : public srf::node::GenericSource<T>
{
  public:
    Reader(std::shared_ptr<grpc::internal::AsyncReaderInterface<T>> reader) : m_reader(std::move(reader))
    {
        CHECK(m_reader);
    }

  private:
    void data_source(rxcpp::subscriber<T>& s) final
    {
        while (s.is_subscribed())
        {
            boost::fibers::promise<bool> promise;
            m_reader->Read(&m_request, &promise);
            auto ok = promise.get_future().get();
            if (!ok)
            {
                return;
            }
            s.on_next(std::move(m_request));
        }
    }

    void on_stop(const rxcpp::subscription& subscription) const final {}

    T m_request;
    const std::shared_ptr<grpc::internal::AsyncReaderInterface<T>> m_reader;
};

}  // namespace srf::internal::rpc
