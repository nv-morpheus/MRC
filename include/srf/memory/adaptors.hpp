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

#include <srf/memory/resources/memory_resource.hpp>

#include <glog/logging.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>  // for size_t
#include <type_traits>

namespace srf::memory {

template <typename Upstream>
class adaptor : public memory_resource
{
  public:
    using reference_type = std::add_lvalue_reference_t<decltype(*std::declval<Upstream>())>;
    using pointer_type   = std::remove_reference_t<decltype(&*std::declval<Upstream>())>;

    adaptor(Upstream upstream) : m_upstream(std::move(upstream)) {}

  protected:
    reference_type upstream()
    {
        CHECK(m_upstream);
        return *m_upstream;
    }

  private:
    memory_kind do_kind() const final
    {
        return m_upstream->kind();
    }

    Upstream m_upstream;
};

class rmm_adaptor final : public memory_resource
{
  public:
    using pointer_type = typename std::add_pointer_t<rmm::mr::device_memory_resource>;

    rmm_adaptor(rmm::mr::device_memory_resource* upstream) : m_upstream(upstream) {}

  private:
    void* do_allocate(std::size_t bytes) final
    {
        return m_upstream->allocate(bytes, rmm::cuda_stream_per_thread);
    }

    void do_deallocate(void* ptr, std::size_t bytes) final
    {
        m_upstream->deallocate(ptr, bytes, rmm::cuda_stream_per_thread);
    }

    memory_kind do_kind() const final
    {
        return memory_kind::device;
    }

    rmm::mr::device_memory_resource* m_upstream;
};

template <template <class> class Resource, typename Upstream, typename... Args>
auto make_shared_resource(Upstream upstream, Args&&... args)
{
    return std::make_shared<Resource<Upstream>>(std::move(upstream), std::forward<Args>(args)...);
}

template <template <class> class Resource, typename Upstream, typename... Args>
auto make_unique_resource(Upstream upstream, Args&&... args)
{
    return std::make_unique<Resource<Upstream>>(std::move(upstream), std::forward<Args>(args)...);
}

}  // namespace srf::memory
