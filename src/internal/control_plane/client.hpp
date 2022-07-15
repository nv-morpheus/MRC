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

#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "srf/runnable/runner.hpp"

#include <grpcpp/completion_queue.h>

namespace srf::internal::control_plane {

class Client : public Service
{
  public:
    Client(runnable::Resources& resources);

    // if we already have an grpc progress engine running, we don't need run another
    Client(runnable::Resources& resources, std::shared_ptr<grpc::CompletionQueue> cq);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    std::shared_ptr<grpc::CompletionQueue> m_cq;

    std::unique_ptr<srf::runnable::Runner> m_progress_engine;
    std::unique_ptr<srf::runnable::Runner> m_event_handler;
};

}  // namespace srf::internal::control_plane
