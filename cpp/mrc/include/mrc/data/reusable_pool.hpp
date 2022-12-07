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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <glog/logging.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace mrc::data {

/**
 * @brief A move-only holder of an object of T that is acquired from a ReusablePool<T> and will be returned to the same
 * pool when it goes out of scope or is released.
 */
template <typename T>
class Reusable;

/**
 * @brief An object that is transformed from a Resusable<T> such that the resulting object is copyable; however, access
 * to the value of T is readonly.
 */
template <typename T>
class SharedReusable;

/**
 * @brief A resource pool which holds upto capacity of unique_ptr<T> which are provided to the requesting callers as
 * Reusable<T> instead of unique_ptr<T>. Reusable objects are returned to the pool when destroyed.
 *
 * An optional on return lambda can be called on a ref of T being returned to the pool. This allows the returning object
 * to be reset to a known state before being added back to the resource pool.
 *
 * Items can be added up to the predefined capacity which must be a power of 2. The add_item and return_item should
 * never block the caller because the channel should never be full. The number of items actively managed by the pool
 * must be less than the capacity.
 *
 * It is possible for the caller of await_item to block on when the pool is empty and all avaiable items are in use.
 *
 * @tparam T
 */
template <typename T>
class ReusablePool final : public std::enable_shared_from_this<ReusablePool<T>>
{
  public:
    using item_t      = std::unique_ptr<T>;
    using on_return_t = std::function<void(T&)>;

    ~ReusablePool()
    {
        // this will prevent items from being returned to the pool
        m_channel.close();
    }

    DELETE_COPYABILITY(ReusablePool);
    DELETE_MOVEABILITY(ReusablePool);

    static std::shared_ptr<ReusablePool<T>> create(std::size_t capacity, on_return_t on_return_fn = nullptr)
    {
        return std::shared_ptr<ReusablePool>(new ReusablePool(capacity, std::move(on_return_fn)));
    }

    void add_item(item_t item)
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        if (m_size + 1 < m_capacity) /* [[likely]] */  // todo(#54) - cpp20
        {
            m_channel.push(std::move(item));
            m_size++;
            return;
        }
        throw std::length_error("pool capacity exceeded");
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
    ReusablePool(std::size_t capacity, on_return_t on_return_fn) :
      m_capacity(capacity),
      m_on_return_fn(std::move(on_return_fn)),
      m_channel(capacity)
    {}

    void return_item(std::unique_ptr<T> item)
    {
        if (m_on_return_fn)
        {
            m_on_return_fn(*item);
        }
        m_channel.push(std::move(item));
    }

    std::mutex m_mutex;
    std::size_t m_size{0UL};
    const std::size_t m_capacity;
    std::function<void(T&)> m_on_return_fn{nullptr};
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

    Reusable(Reusable&&) noexcept            = default;
    Reusable& operator=(Reusable&&) noexcept = default;

    DELETE_COPYABILITY(Reusable);

    ~Reusable()
    {
        release();
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

    void release()
    {
        if (m_data)
        {
            m_pool->return_item(std::move(m_data));
        }
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
                 pool->return_item(std::move(unique));
             }),
      m_pool(std::move(pool))
    {}

  public:
    SharedReusable() = default;
    SharedReusable(Reusable<T>&& data) : SharedReusable(std::move(data.m_data), data.m_pool) {}

    SharedReusable(const SharedReusable&)            = default;
    SharedReusable& operator=(const SharedReusable&) = default;

    SharedReusable(SharedReusable&&) noexcept            = default;
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

    void release()
    {
        m_data.reset();
    }

  private:
    std::shared_ptr<const T> m_data;
    std::shared_ptr<pool_t> m_pool;

    friend pool_t;
};

}  // namespace mrc::data
