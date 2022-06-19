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

#include "srf/channel/buffered_channel.hpp"
#include "srf/utils/macros.hpp"

#include <glog/logging.h>

#include <atomic>
#include <boost/fiber/buffered_channel.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace srf::data {

template <typename T>
class Reusable;

template <typename T>
class SharedReusable;

template <typename T>
class ReusablePool final : public std::enable_shared_from_this<ReusablePool<T>>
{
    ReusablePool(std::size_t capacity) : m_size(0), m_capacity(capacity), m_channel(capacity) {}

  public:
    using item_t = std::unique_ptr<T>;

    DELETE_COPYABILITY(ReusablePool);
    DELETE_MOVEABILITY(ReusablePool);

    static std::shared_ptr<ReusablePool<T>> create(std::size_t capacity)
    {
        return std::shared_ptr<ReusablePool>(new ReusablePool(capacity));
    }

    void add_item(item_t item)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        if (m_size >= m_capacity)
        {
            throw std::length_error("pool capacity exceeded");
        }
        m_channel.push(std::move(item));
        m_size++;
    }

    template <typename... ArgsT>
    void emplace(ArgsT&&... args)
    {
        add_item(std::make_unique<T>(std::forward<ArgsT>(args)...));
    }

    Reusable<T> await_item()
    {
        item_t item;
        m_channel.pop(item);
        return Reusable<T>(std::move(item), this->shared_from_this());
    }

    /**
     * @brief Number of items managed by the pool
     */
    std::size_t size() const
    {
        return m_size;
    }

  private:
    std::mutex m_mutex;
    std::size_t m_size;
    const std::size_t m_capacity;
    boost::fibers::buffered_channel<item_t> m_channel;

    friend Reusable<T>;
    friend SharedReusable<T>;
};

template <typename T>
class Reusable final
{
    using pool_t = ReusablePool<T>;

    Reusable(std::unique_ptr<T> data, std::shared_ptr<pool_t> pool) : m_data(std::move(data)), m_pool(std::move(pool))
    {}

  public:
    Reusable() = default;

    Reusable(Reusable&&) noexcept = default;
    Reusable& operator=(Reusable&&) noexcept = default;

    DELETE_COPYABILITY(Reusable);

    ~Reusable()
    {
        if (m_data)
        {
            m_pool->m_channel.push(std::move(m_data));
        }
    }

    T& operator*()
    {
        CHECK(m_data);
        return *m_data;
    }

    T* operator->()
    {
        CHECK(m_data);
        return m_data.get();
    }

  private:
    std::unique_ptr<T> m_data;
    std::shared_ptr<pool_t> m_pool;

    friend pool_t;
    friend SharedReusable<T>;
};

template <typename T>
class SharedReusable final
{
    using pool_t = ReusablePool<T>;

    SharedReusable(std::unique_ptr<T> data, std::shared_ptr<pool_t> pool) :
      m_data(data.release(),
             [pool](T* ptr) {
                 std::unique_ptr<T> unique(ptr);
                 pool->m_channel.push(std::move(unique));
             }),
      m_pool(std::move(pool))
    {}

  public:
    SharedReusable() = default;
    SharedReusable(Reusable<T>&& data) : SharedReusable(std::move(data.m_data), data.m_pool) {}

    SharedReusable(const SharedReusable&) = default;
    SharedReusable& operator=(const SharedReusable&) = default;

    SharedReusable(SharedReusable&&) noexcept = default;
    SharedReusable& operator=(SharedReusable&&) noexcept = default;

    ~SharedReusable() = default;

    SharedReusable& operator=(Reusable<T>&& rhs)
    {
        *this = SharedReusable(std::move(rhs));
        return *this;
    }

    const T& operator*() const
    {
        CHECK(m_data);
        return *m_data;
    }

    const T* operator->() const
    {
        CHECK(m_data);
        return m_data.get();
    }

  private:
    std::shared_ptr<const T> m_data;
    std::shared_ptr<pool_t> m_pool;

    friend pool_t;
};

}  // namespace srf::data
