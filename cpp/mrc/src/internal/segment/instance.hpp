/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/service.hpp"

#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::internal::runtime {
class Partition;
}

namespace mrc::internal::pipeline {
class Resources;
}  // namespace mrc::internal::pipeline
namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::internal::segment {
class Definition;
class Builder;

// todo(ryan) - inherit from service
class Instance final : public Service
{
  public:
    Instance(std::shared_ptr<const Definition> definition, SegmentRank rank, runtime::Partition& partition);
    ~Instance() override;

    const std::string& name() const;
    const SegmentID& id() const;
    const SegmentRank& rank() const;
    const SegmentAddress& address() const;

    std::shared_ptr<manifold::Interface> create_manifold(const PortName& name);
    void attach_manifold(std::shared_ptr<manifold::Interface> manifold);

  protected:
    const std::string& info() const;

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    void callback_on_state_change(const std::string& name, const mrc::runnable::Runner::State& new_state);

    std::string m_name;
    SegmentID m_id;
    SegmentRank m_rank;
    SegmentAddress m_address;
    std::string m_info;

    std::unique_ptr<Builder> m_builder;
    runtime::Partition& m_partition_runtime;
    const std::size_t m_default_partition_id;

    std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_runners;
    std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_egress_runners;
    std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_ingress_runners;

    mutable std::mutex m_mutex;
};

}  // namespace mrc::internal::segment
