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

#include "internal/ucx/worker.hpp"

#include "srf/channel/status.hpp"
#include "srf/node/generic_sink.hpp"
#include "srf/runnable/context.hpp"

#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>

#include <memory>
#include <utility>

namespace srf::internal::data_plane {

class DataPlaneClientWorker : public node::GenericSink<void*>
{
  public:
    DataPlaneClientWorker(std::shared_ptr<ucx::Worker> worker) : m_worker(std::move(worker)) {}

  private:
    void on_data(void*&& data) final;

    std::shared_ptr<ucx::Worker> m_worker;
};

}  // namespace srf::internal::data_plane
