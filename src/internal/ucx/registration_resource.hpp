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

#pragma once

#include "internal/ucx/registration_cache.hpp"

#include "srf/cuda/common.hpp"
#include "srf/cuda/device_guard.hpp"
#include "srf/memory/adaptors.hpp"

#include <memory>

namespace srf::internal::ucx {

/**
 * @brief Memory Resource adaptor to provide UCX registration to allocated blocks.
 *
 * This is an internal class and used only for constructing device memory resources. A more general implementation might
 * separate our the CUDA DeviceID requirement.
 *
 * @tparam PointerT
 */
template <typename PointerT>
class RegistrationResource : public srf::memory::adaptor<PointerT>
{
  public:
    RegistrationResource(PointerT upstream, std::shared_ptr<RegistrationCache> registration_cache, int cuda_device_id) :
      srf::memory::adaptor<PointerT>(std::move(upstream)),
      m_registration_cache(std::move(registration_cache)),
      m_cuda_device_id(cuda_device_id)
    {
        CHECK(m_registration_cache);
    }

    const RegistrationCache& registration_cache() const
    {
        return *m_registration_cache;
    }

  private:
    void* do_allocate(std::size_t bytes) final
    {
        DeviceGuard guard(m_cuda_device_id);
        auto* ptr = this->resource().allocate(bytes);
        m_registration_cache->add_block(ptr, bytes);
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes) final
    {
        DeviceGuard guard(m_cuda_device_id);
        auto size = m_registration_cache->drop_block(ptr, bytes);
        this->resource().deallocate(ptr, size);
    }

    const std::shared_ptr<RegistrationCache> m_registration_cache;
    const int m_cuda_device_id;
};

}  // namespace srf::internal::ucx
