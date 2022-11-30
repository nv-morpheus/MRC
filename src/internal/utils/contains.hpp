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

#include <iterator>

namespace mrc {

template <typename Container, typename Key>
bool contains(const Container& container, const Key& key)
{
    auto search = container.find(key);
    return (search == container.end() ? false : true);
}

template <typename C>
class KeyIterator : public std::iterator<std::bidirectional_iterator_tag,
                                         typename C::key_type,
                                         typename C::difference_type,
                                         typename C::pointer,
                                         typename C::reference>
{
  public:
    KeyIterator() = default;
    explicit KeyIterator(typename C::const_iterator it) : m_iter(it) {}

    const typename C::key_type& operator*() const
    {
        return m_iter->first;
    }
    const typename C::key_type* operator->() const
    {
        return &m_iter->first;
    }

    KeyIterator& operator++()
    {
        ++m_iter;
        return *this;
    }
    KeyIterator operator++(int)
    {
        KeyIterator it(*this);
        ++*this;
        return it;
    }

    KeyIterator& operator--()
    {
        --m_iter;
        return *this;
    }
    KeyIterator operator--(int)
    {
        KeyIterator it(*this);
        --*this;
        return it;
    }

    friend bool operator==(const KeyIterator& lhs, const KeyIterator& rhs)
    {
        return lhs.m_iter == rhs.m_iter;
    }

    friend bool operator!=(const KeyIterator& lhs, const KeyIterator& rhs)
    {
        return !(lhs == rhs);
    }

  private:
    typename C::const_iterator m_iter;
};

template <typename C>
KeyIterator<C> begin_keys(const C& c)
{
    return KeyIterator<C>(c.begin());
}

template <typename C>
KeyIterator<C> end_keys(const C& c)
{
    return KeyIterator<C>(c.end());
}

}  // namespace mrc
