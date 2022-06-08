/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <srf/memory/adaptors.hpp>
#include <srf/memory/resources/host/pinned_memory_resource.hpp>

#include <cuda/memory_resource>

namespace srf::memory {

using numa_node_id_t = std::uint32_t;

// this allows for host or pinned host memory, but forbids managed memory
struct host_view
{
    using view_type = cuda::resource_view<cuda::memory_access::host, cuda::memory_location::host, cuda::resident>;
    static cuda::memory_resource<cuda::memory_kind::pinned>* initial_resource()
    {
        static pinned_memory_resource mr{};
        return &mr;
    }
};

// this allows for device memory, but forbids managed memory or pinned memory
/*
struct device_view
{
    using device_view_type =
        cuda::resource_view<cuda::memory_access::device, cuda::memory_location::device, cuda::resident>;
    static cuda::memory_resource<cuda::memory_kind::pinned>* initial_resource()
    {
        static pinned_memory_resource mr{};
        return &mr;
    }
};
*/

namespace detail {

template <typename ViewT>
class thread_local_view : public ViewT
{
    using view_type = typename ViewT::view_type;

  public:
    static void set(view_type view)
    {
        m_view = view;
    }

    static view_type& get()
    {
        return m_view;
    }

  private:
    thread_local static view_type m_view;
};

template <>
inline thread_local typename host_view::view_type thread_local_view<host_view>::m_view = host_view::initial_resource();

}  // namespace detail

inline void set_thread_host_resource(typename host_view::view_type view)
{
    detail::thread_local_view<host_view>::set(view);
}

inline typename host_view::view_type get_thread_host_resource()
{
    return detail::thread_local_view<host_view>::get();
}

namespace detail {
template <typename Pointer>
class polymorphic_storage
{
  public:
    polymorphic_storage(Pointer ptr) : m_pmr(std::move(ptr)){};
    ~polymorphic_storage() = default;

    std::pmr::memory_resource* pmr() noexcept
    {
        return &m_pmr;
    }

  private:
    polymorphic_adaptor<Pointer> m_pmr;
};
}  // namespace detail

template <typename T>
class host_allocator : private detail::polymorphic_storage<typename host_view::view_type>,
                       public std::pmr::polymorphic_allocator<T>
{
    using view_type   = typename host_view::view_type;
    using storage_t   = detail::polymorphic_storage<typename host_view::view_type>;
    using allocator_t = std::pmr::polymorphic_allocator<T>;

  public:
    host_allocator(view_type mr = get_thread_host_resource()) : storage_t(std::move(mr)), allocator_t(storage_t::pmr())
    {}
    ~host_allocator() = default;
};

}  // namespace srf::memory
