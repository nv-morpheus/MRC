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

#include "srf/memory/resources/memory_resource.hpp"
#include "srf/type_traits.hpp"

#include <cuda/memory_resource>
#include <glog/logging.h>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>

namespace srf::memory {

struct polymorphic_resource_view
{
    virtual ~polymorphic_resource_view() = default;
    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
    {
        return do_allocate(bytes, alignment);
    }
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
    {
        do_deallocate(ptr, bytes, alignment);
    }

    memory_kind_type kind() const
    {
        return do_kind();
    }

  private:
    virtual void* do_allocate(std::size_t, std::size_t)         = 0;
    virtual void do_deallocate(void*, std::size_t, std::size_t) = 0;
    virtual memory_kind_type do_kind() const                    = 0;
};

namespace detail {

template <typename... Properties>
struct resource_view_interface : public polymorphic_resource_view
{
    ~resource_view_interface() override                              = default;
    virtual const ::cuda::resource_view<Properties...>& view() const = 0;
};

template <typename... Properties>
class resource_view_raw_storage final : public resource_view_interface<Properties...>
{
  public:
    template <typename Kind>
    explicit resource_view_raw_storage(memory_resource<Kind>* mr) : m_view(mr), m_kind(mr->kind())
    {}
    ~resource_view_raw_storage() override = default;

    const ::cuda::resource_view<Properties...>& view() const final
    {
        return m_view;
    }

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) final
    {
        return m_view->allocate(bytes, alignment);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) final
    {
        m_view->deallocate(ptr, bytes, alignment);
    }

    memory_kind_type do_kind() const final
    {
        return m_kind;
    }

    ::cuda::resource_view<Properties...> m_view;
    memory_kind_type m_kind;
};

template <typename ResourcePointer, typename... Properties>
class resource_view_smart_storage final : public resource_view_interface<Properties...>
{
  public:
    resource_view_smart_storage(ResourcePointer pointer) :
      m_storage(pointer),
      m_view(pointer.get()),
      m_kind(pointer->kind())
    {}
    ~resource_view_smart_storage() override = default;

    const ::cuda::resource_view<Properties...>& view() const final
    {
        return m_view;
    }

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) final
    {
        return m_view->allocate(bytes, alignment);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) final
    {
        m_view->deallocate(ptr, bytes, alignment);
    }

    memory_kind_type do_kind() const final
    {
        return m_kind;
    }

    ResourcePointer m_storage{nullptr};
    ::cuda::resource_view<Properties...> m_view;
    memory_kind_type m_kind;
};

}  // namespace detail

template <typename... Properties>
class resource_view final
{
  public:
    resource_view() = delete;

    template <typename ResourcePointer, typename = std::enable_if_t<std::is_pointer_v<ResourcePointer>>>
    resource_view(ResourcePointer pointer) :
      m_storage(std::make_shared<detail::resource_view_raw_storage<Properties...>>(std::move(pointer)))
    {}

    template <typename ResourcePointer,
              typename = std::enable_if_t<is_shared_ptr<ResourcePointer>::value>,
              typename = void>
    resource_view(ResourcePointer pointer) :
      m_storage(
          std::make_shared<detail::resource_view_smart_storage<ResourcePointer, Properties...>>(std::move(pointer)))
    {}

    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
    {
        return m_storage->allocate(bytes, alignment);
    }

    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
    {
        m_storage->deallocate(ptr, bytes, alignment);
    }

    memory_kind_type kind() const
    {
        return m_storage->kind();
    }

    std::shared_ptr<polymorphic_resource_view> prv() const
    {
        return m_storage;
    }

  private:
    std::shared_ptr<detail::resource_view_interface<Properties...>> m_storage;
};

}  // namespace srf::memory
