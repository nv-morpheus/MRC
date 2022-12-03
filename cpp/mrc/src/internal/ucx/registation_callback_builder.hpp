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

#include "internal/memory/callback_adaptor.hpp"
#include "internal/ucx/registration_cache.hpp"

#include <glog/logging.h>

#include <memory>
#include <mutex>

namespace mrc::internal::ucx {

class RegistrationCallbackBuilder final : public memory::CallbackBuilder
{
  public:
    void add_registration_cache(std::shared_ptr<RegistrationCache> registration_cache)
    {
        register_callbacks(
            [registration_cache](void* addr, std::size_t bytes) { registration_cache->add_block(addr, bytes); },
            [registration_cache](void* addr, std::size_t bytes) { registration_cache->drop_block(addr, bytes); });
    }

  private:
    using CallbackBuilder::register_callbacks;
};

}  // namespace mrc::internal::ucx
