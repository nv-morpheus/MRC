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

#include "mrc/runnable/internal_service.hpp"
#include "mrc/runnable/launch_options.hpp"

#include <map>

namespace mrc {

class ServiceOptions
{
  public:
    void set_service_options(runnable::InternalServiceType service_type, const runnable::LaunchOptions& launch_options);
    void set_default_options(const runnable::LaunchOptions& launch_options);

    const runnable::LaunchOptions& service_options(runnable::InternalServiceType service_type) const;
    const runnable::LaunchOptions& default_options() const;

  private:
    std::map<runnable::InternalServiceType, runnable::LaunchOptions> m_service_options;
    runnable::LaunchOptions m_default_options;
};

}  // namespace mrc
