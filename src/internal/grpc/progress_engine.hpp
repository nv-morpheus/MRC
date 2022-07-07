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

#include "srf/node/generic_source.hpp"

#include <boost/fiber/operations.hpp>
#include <grpc/support/time.h>
#include <grpcpp/completion_queue.h>

#include <chrono>

namespace srf::internal::rpc {

struct ProgressEvent
{
    void* tag;
    bool ok;
};

class ProgressEngine final : public srf::node::GenericSource<ProgressEvent>
{
  public:
    ProgressEngine(std::shared_ptr<grpc::CompletionQueue> cq);

  private:
    void data_source(rxcpp::subscriber<ProgressEvent>& s) final;

    // disabling stop on this source
    // the proper way to stop this source is to issue a CompletionQueue::Shutdown()
    void on_stop(const rxcpp::subscription& subscription) const final;

    std::shared_ptr<grpc::CompletionQueue> m_cq;
};

}  // namespace srf::internal::rpc
