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

#include "mrc/options/services.hpp"

#include "mrc/runnable/internal_service.hpp"
#include "mrc/runnable/launch_options.hpp"

#include <utility>

namespace mrc {

void ServiceOptions::set_service_options(runnable::InternalServiceType service_type,
                                         const runnable::LaunchOptions& launch_options)
{
    m_service_options[service_type] = launch_options;
}

void ServiceOptions::set_default_options(const runnable::LaunchOptions& launch_options)
{
    m_default_options = launch_options;
}

const runnable::LaunchOptions& ServiceOptions::service_options(runnable::InternalServiceType service_type) const
{
    auto search = m_service_options.find(service_type);
    if (search == m_service_options.end())
    {
        return m_default_options;
    }
    return search->second;
}
const runnable::LaunchOptions& ServiceOptions::default_options() const
{
    return m_default_options;
}

}  // namespace mrc
