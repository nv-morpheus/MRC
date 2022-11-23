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

#include "internal/grpc/stream_writer.hpp"

#include "mrc/protos/architect.pb.h"
#include "mrc/utils/macros.hpp"

#include <string>

namespace mrc::internal::control_plane::server {

class ClientInstance
{
  public:
    using instance_id_t = std::uint64_t;
    using writer_t      = std::shared_ptr<rpc::StreamWriter<mrc::protos::Event>>;

    ClientInstance(writer_t writer, std::string worker_address) :
      m_stream_writer(std::move(writer)),
      m_worker_address(std::move(worker_address))
    {
        // CHECK(m_stream_writer);
    }

    DELETE_MOVEABILITY(ClientInstance);
    DELETE_COPYABILITY(ClientInstance);

    instance_id_t get_id() const
    {
        return reinterpret_cast<instance_id_t>(this);
    }

    rpc::StreamWriter<mrc::protos::Event>& stream_writer() const
    {
        return *m_stream_writer;
    }

    const std::string& worker_address() const
    {
        return m_worker_address;
    }

  private:
    const std::shared_ptr<rpc::StreamWriter<mrc::protos::Event>> m_stream_writer;
    const std::string m_worker_address;
};

}  // namespace mrc::internal::control_plane::server
