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

#include <cuda/memory_resource>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>  // for size_t

namespace srf::memory {

template <typename Upstream>
class upstream_resource
  : public memory_resource<typename std::remove_reference_t<decltype(*std::declval<Upstream>())>::memory_kind>
{
    using kind_type = typename std::remove_reference_t<decltype(*std::declval<Upstream>())>::memory_kind;

  public:
    using cuda_resource_type = ::cuda::memory_resource<kind_type>;
    using resource_type      = memory_resource<kind_type>;
    using pointer_type       = std::remove_reference_t<decltype(&*std::declval<Upstream>())>;

    upstream_resource(Upstream upstream, std::string tag) : resource_type(tag), m_upstream(std::move(upstream))
    {
        add_tags(resource()->tags());
    }
    ~upstream_resource() override = default;

  protected:
    pointer_type resource() const
    {
        return &*m_upstream;
    }

    bool do_is_equal(const cuda_resource_type& other) const noexcept override
    {
        return m_upstream->is_equal(other);
    }

  private:
    memory_kind_type do_kind() const final
    {
        return m_upstream->kind();
    }

    using resource_type::add_tag;
    using resource_type::add_tags;

    Upstream m_upstream;
    std::vector<std::string> m_tags;
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

class rmm_adaptor : public memory_resource<::cuda::memory_kind::with_properties<::cuda::memory_access::device>>
{
    using kind_type = ::cuda::memory_kind::with_properties<::cuda::memory_access::device>;

  public:
    using cuda_resource_type = ::cuda::memory_resource<kind_type>;
    using resource_type      = memory_resource<kind_type>;
    using pointer_type       = const resource_type*;

    rmm_adaptor(rmm::mr::device_memory_resource* upstream, std::string tag = "rmm upstream") :
      resource_type(tag),
      m_upstream(std::move(upstream))
    {
        add_tags(resource()->tags());
    }
    ~rmm_adaptor() override = default;

  protected:
    pointer_type resource() const
    {
        return this;
    }

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        return m_upstream->allocate(bytes, rmm::cuda_stream_per_thread);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override
    {
        m_upstream->deallocate(ptr, bytes, rmm::cuda_stream_per_thread);
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::device;
    }

    using resource_type::add_tag;
    using resource_type::add_tags;

    rmm::mr::device_memory_resource* m_upstream;
    std::vector<std::string> m_tags;
};

template <typename... Properties>
class view_adaptor : public ::cuda::memory_resource<::cuda::memory_kind::with_properties<Properties...>>
{
  public:
    using resource_type = ::cuda::memory_resource<::cuda::memory_kind::with_properties<Properties...>>;
    using view_type     = ::cuda::resource_view<Properties...>;

    view_adaptor(view_type view) : m_view(std::move(view)) {}
    ~view_adaptor() override = default;

  protected:
    view_type& view()
    {
        return m_view;
    }

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        return m_view->allocate(bytes, alignment);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override
    {
        return m_view->deallocate(ptr, bytes, alignment);
    }

    bool do_is_equal(const resource_type& other) const noexcept override
    {
        return m_view->is_equal(other);
    }

    view_type m_view;
};

namespace detail {

class __polymorphic_adaptor_base : public _LIBCUDACXX_STD_PMR_NS::memory_resource
{
  public:
    virtual ::cuda::resource_view<::cuda::memory_access::host> view() const noexcept = 0;
};

}  // namespace detail

template <typename _Pointer>
class polymorphic_adaptor final : public detail::__polymorphic_adaptor_base
{
    using resource_type = std::remove_reference_t<decltype(*std::declval<_Pointer>())>;

    static constexpr bool __is_host_accessible_resource =  // NOLINT
        ::cuda::has_property<resource_type, ::cuda::memory_access::host>::value;
    static_assert(__is_host_accessible_resource,
                  "Pointer must be a pointer-like type to a type that allocates host-accessible memory.");

  public:
    polymorphic_adaptor(_Pointer __mr) : __mr_{std::move(__mr)} {}

    using raw_pointer = std::remove_reference_t<decltype(&*std::declval<_Pointer>())>;

    raw_pointer memory_resource() const noexcept
    {
        return &*__mr_;
    }

    ::cuda::resource_view<::cuda::memory_access::host> view() const noexcept override
    {
        return ::cuda::resource_view<::cuda::memory_access::host>(&*__mr_);
    }

  private:
    void* do_allocate(std::size_t __bytes, std::size_t __alignment) override
    {
        return __mr_->allocate(__bytes, __alignment);
    }

    void do_deallocate(void* __p, std::size_t __bytes, std::size_t __alignment) override
    {
        return __mr_->deallocate(__p, __bytes, __alignment);
    }

    bool do_is_equal(_LIBCUDACXX_STD_PMR_NS::memory_resource const& __other) const noexcept override
    {
        const auto* __other_p = dynamic_cast<detail::__polymorphic_adaptor_base const*>(&__other);
        return __other_p and (__other_p->view() == view());
    }

    _Pointer __mr_;  // NOLINT
};

template <typename... Properties>
class polymorphic_adaptor<::cuda::resource_view<Properties...>> final : public detail::__polymorphic_adaptor_base
{
    static_assert(std::bool_constant<(std::is_same<::cuda::memory_access::host, Properties>{} || ...)>::value,
                  "must be a host accessible view");

  public:
    polymorphic_adaptor(::cuda::resource_view<Properties...> view) : m_view(std::move(view)) {}

    ::cuda::resource_view<::cuda::memory_access::host> view() const noexcept override
    {
        return m_view;
    }

  private:
    void* do_allocate(std::size_t __bytes, std::size_t __alignment) override
    {
        return m_view->allocate(__bytes, __alignment);
    }

    void do_deallocate(void* __p, std::size_t __bytes, std::size_t __alignment) override
    {
        return m_view->deallocate(__p, __bytes, __alignment);
    }

    bool do_is_equal(_LIBCUDACXX_STD_PMR_NS::memory_resource const& __other) const noexcept override
    {
        const auto* __other_p = dynamic_cast<detail::__polymorphic_adaptor_base const*>(&__other);
        return __other_p and (__other_p->view() == m_view);
    }

    ::cuda::resource_view<Properties...> m_view;
};

}  // namespace srf::memory
