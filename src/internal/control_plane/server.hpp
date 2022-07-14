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

#include "internal/grpc/server.hpp"

#include <memory>
#include <string>

namespace srf::internal::control_plane {

class ServerResources;

class Server
{
  public:
    Server(std::string url);
    Server(int port);
    ~Server() = default;

    void shutdown();

  private:
    std::unique_ptr<rpc::Server> m_server;
    std::shared_ptr<ServerResources> m_resources;
};

}  // namespace srf::internal::control_plane
