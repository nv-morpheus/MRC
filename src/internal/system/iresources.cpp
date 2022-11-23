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

#include "mrc/engine/system/iresources.hpp"

#include "internal/system/resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/topology.hpp"

#include "mrc/engine/system/isystem.hpp"

#include <memory>
#include <utility>

namespace mrc::internal::system {

IResources::IResources(std::shared_ptr<ISystem> system) :
  m_impl(std::make_unique<Resources>(SystemProvider(System::unwrap(*system))))
{}
IResources::~IResources() = default;

void IResources::add_thread_initializer(std::function<void()> initializer_fn)
{
    m_impl->register_thread_local_initializer(m_impl->system().topology().cpu_set(), std::move(initializer_fn));
}

void IResources::add_thread_finalizer(std::function<void()> finalizer_fn)
{
    m_impl->register_thread_local_finalizer(m_impl->system().topology().cpu_set(), std::move(finalizer_fn));
}

}  // namespace mrc::internal::system
