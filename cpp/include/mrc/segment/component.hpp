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

#include "mrc/segment/object.hpp"

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <utility>

namespace mrc::segment {

template <typename ResourceT>
class Component final : public Object<ResourceT>
{
  public:
    Component(std::unique_ptr<ResourceT> resource) : m_resource(std::move(resource)) {}
    ~Component() final = default;

  private:
    ResourceT* get_object() const final
    {
        CHECK(m_resource);
        return m_resource.get();
    }
    std::unique_ptr<ResourceT> m_resource;
};

}  // namespace mrc::segment
