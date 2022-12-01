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

#include "mrc/engine/segment/ibuilder.hpp"

#include "internal/segment/builder.hpp"

#include "mrc/segment/object.hpp"

#include <glog/logging.h>

#include <utility>

namespace mrc::internal::segment {

IBuilder::IBuilder(Builder* impl) : m_impl(impl)
{
    CHECK(m_impl != nullptr);
}

IBuilder::~IBuilder() = default;

const std::string& IBuilder::name() const
{
    CHECK(m_impl);
    return m_impl->name();
}

bool IBuilder::has_object(const std::string& name) const
{
    CHECK(m_impl);
    return m_impl->has_object(name);
}

mrc::segment::ObjectProperties& IBuilder::find_object(const std::string& name)
{
    CHECK(m_impl);
    return m_impl->find_object(name);
}

void IBuilder::add_object(const std::string& name, std::shared_ptr<::mrc::segment::ObjectProperties> object)
{
    CHECK(m_impl);
    return m_impl->add_object(name, std::move(object));
}

void IBuilder::add_runnable(const std::string& name, std::shared_ptr<mrc::runnable::Launchable> runnable)
{
    CHECK(m_impl);
    return m_impl->add_runnable(name, std::move(runnable));
}

std::shared_ptr<mrc::segment::IngressPortBase> IBuilder::get_ingress_base(const std::string& name)
{
    CHECK(m_impl);
    return m_impl->get_ingress_base(name);
}

std::shared_ptr<mrc::segment::EgressPortBase> IBuilder::get_egress_base(const std::string& name)
{
    CHECK(m_impl);
    return m_impl->get_egress_base(name);
}

std::function<void(std::int64_t)> IBuilder::make_throughput_counter(const std::string& name)
{
    CHECK(m_impl);
    return m_impl->make_throughput_counter(name);
}

}  // namespace mrc::internal::segment
