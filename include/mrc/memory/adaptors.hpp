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

#include "mrc/memory/resources/memory_resource.hpp"

#include <glog/logging.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>  // for size_t
#include <type_traits>

namespace mrc::memory {

template <typename PointerT>
class adaptor : public memory_resource
{
  public:
    using reference_type = std::add_lvalue_reference_t<decltype(*std::declval<PointerT>())>;
    using pointer_type   = std::remove_reference_t<decltype(&*std::declval<PointerT>())>;

    adaptor(PointerT upstream) : m_upstream(std::move(upstream)) {}

    reference_type resource()
    {
        CHECK(m_upstream);
        return *m_upstream;
    }

  private:
    memory_kind do_kind() const final
    {
        return m_upstream->kind();
    }

    PointerT m_upstream;
};

class rmm_adaptor : public memory_resource
{
  public:
    virtual rmm::mr::device_memory_resource& rmm_memory_resource() = 0;

  private:
    void* do_allocate(std::size_t bytes) final
    {
        return rmm_memory_resource().allocate(bytes, rmm::cuda_stream_per_thread);
    }

    void do_deallocate(void* ptr, std::size_t bytes) final
    {
        rmm_memory_resource().deallocate(ptr, bytes, rmm::cuda_stream_per_thread);
    }

    memory_kind do_kind() const final
    {
        return memory_kind::device;
    }
};

template <typename PointerT>
class rmm_adaptor_typed final : public rmm_adaptor
{
  public:
    using reference_type = std::add_lvalue_reference_t<decltype(*std::declval<PointerT>())>;
    using pointer_type   = std::remove_reference_t<decltype(&*std::declval<PointerT>())>;

    rmm_adaptor_typed(PointerT upstream) : m_upstream(upstream) {}

    // rmm::mr::device_memory_resource& rmm_memory_resource() final
    reference_type rmm_memory_resource() final
    {
        CHECK(m_upstream);
        return *m_upstream;
    }

  private:
    PointerT m_upstream;
};

template <template <class> class Resource, typename PointerT, typename... Args>
auto make_shared_resource(PointerT upstream, Args&&... args)
{
    return std::make_shared<Resource<PointerT>>(std::move(upstream), std::forward<Args>(args)...);
}

template <template <class> class Resource, typename PointerT, typename... Args>
auto make_unique_resource(PointerT upstream, Args&&... args)
{
    return std::make_unique<Resource<PointerT>>(std::move(upstream), std::forward<Args>(args)...);
}

}  // namespace mrc::memory
