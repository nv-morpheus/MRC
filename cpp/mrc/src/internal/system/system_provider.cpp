/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/system_provider.hpp"

#include <glog/logging.h>

#include <utility>

namespace mrc::internal::system {

SystemProvider::SystemProvider(std::shared_ptr<const System> system) : m_system(std::move(system))
{
    CHECK(m_system);
}
const System& mrc::internal::system::SystemProvider::system() const
{
    DCHECK(m_system);
    return *m_system;
}

}  // namespace mrc::internal::system
