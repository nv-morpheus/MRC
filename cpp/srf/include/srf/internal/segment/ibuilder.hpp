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

#include <srf/runnable/forward.hpp>
#include <srf/types.hpp>

#include <functional>
#include <memory>
#include <string>

// todo(ryan) - most base classes that will be owned by the engine will need to be moved to engine api/lib
namespace srf::segment {

class ObjectProperties;
class EgressPortBase;
class IngressPortBase;

}  // namespace srf::segment

namespace srf::internal::segment {

struct IBuilder
{
    virtual ~IBuilder() = default;

    virtual const std::string& name() const                                                                    = 0;
    virtual bool has_object(const std::string& name) const                                                     = 0;
    virtual ::srf::segment::ObjectProperties& find_object(const std::string& name)                             = 0;
    virtual void add_object(const std::string& name, std::shared_ptr<::srf::segment::ObjectProperties> object) = 0;
    virtual void add_runnable(const std::string& name, std::shared_ptr<runnable::Launchable> runnable)         = 0;
    virtual std::shared_ptr<::srf::segment::IngressPortBase> get_ingress_base(const std::string& name)         = 0;
    virtual std::shared_ptr<::srf::segment::EgressPortBase> get_egress_base(const std::string& name)           = 0;
    virtual std::function<void(std::int64_t)> make_throughput_counter(const std::string& name)                 = 0;
};

}  // namespace srf::internal::segment
