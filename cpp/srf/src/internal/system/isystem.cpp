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

#include "srf/internal/system/isystem.hpp"
#include "internal/system/system.hpp"
#include "internal/system/topology.hpp"
#include "srf/options/options.hpp"

#include <utility>

namespace srf::internal::system {

ISystem::ISystem(std::shared_ptr<Options> options) : m_impl(make_system(std::move(options))) {}
ISystem::~ISystem() = default;

void ISystem::add_thread_initializer(std::function<void()> initializer_fn)
{
    m_impl->register_thread_local_initializer(m_impl->topology().cpu_set(), std::move(initializer_fn));
}

void ISystem::add_thread_finalizer(std::function<void()> finalizer_fn)
{
    m_impl->register_thread_local_finalizer(m_impl->topology().cpu_set(), std::move(finalizer_fn));
}

}  // namespace srf::internal::system
