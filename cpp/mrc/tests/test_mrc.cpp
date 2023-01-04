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

#include "test_mrc.hpp"

#include <utility>

namespace mrc {

std::atomic<int> CopyMoveCounter::global_default_constructed_count = 0;
std::atomic<int> CopyMoveCounter::global_value_constructed_count   = 0;
std::atomic<int> CopyMoveCounter::global_copy_constructed_count    = 0;
std::atomic<int> CopyMoveCounter::global_move_constructed_count    = 0;
std::atomic<int> CopyMoveCounter::global_copy_assignment_count     = 0;
std::atomic<int> CopyMoveCounter::global_move_assignment_count     = 0;

CopyMoveCounter::CopyMoveCounter()
{
    global_default_constructed_count++;
}

CopyMoveCounter::CopyMoveCounter(int value) : m_value(value)
{
    global_value_constructed_count++;
}

CopyMoveCounter::CopyMoveCounter(const CopyMoveCounter& o)
{
    m_value = o.m_value;

    o.m_was_copied = true;

    m_copy_constructed_count = o.m_copy_constructed_count + 1;
    m_move_constructed_count = o.m_move_constructed_count;
    m_copy_assignment_count  = o.m_copy_assignment_count;
    m_move_assignment_count  = o.m_move_assignment_count;

    global_copy_constructed_count++;
}
CopyMoveCounter::CopyMoveCounter(CopyMoveCounter&& o)
{
    std::swap(m_value, o.m_value);

    o.m_was_moved = true;

    m_copy_constructed_count = o.m_copy_constructed_count;
    m_move_constructed_count = o.m_move_constructed_count + 1;
    m_copy_assignment_count  = o.m_copy_assignment_count;
    m_move_assignment_count  = o.m_move_assignment_count;

    global_move_constructed_count++;
}

CopyMoveCounter& CopyMoveCounter::operator=(const CopyMoveCounter& o)
{
    m_value = o.m_value;

    o.m_was_copied = true;

    m_copy_constructed_count = o.m_copy_constructed_count;
    m_move_constructed_count = o.m_move_constructed_count;
    m_copy_assignment_count  = o.m_copy_assignment_count + 1;
    m_move_assignment_count  = o.m_move_assignment_count;

    global_copy_assignment_count++;

    return *this;
}

CopyMoveCounter& CopyMoveCounter::operator=(CopyMoveCounter&& o)
{
    std::swap(m_value, o.m_value);

    o.m_was_moved = true;

    m_copy_constructed_count = o.m_copy_constructed_count;
    m_move_constructed_count = o.m_move_constructed_count;
    m_copy_assignment_count  = o.m_copy_assignment_count;
    m_move_assignment_count  = o.m_move_assignment_count + 1;

    global_move_assignment_count++;

    return *this;
}

std::size_t CopyMoveCounter::copy_constructed_count() const
{
    return m_copy_constructed_count;
}

std::size_t CopyMoveCounter::copy_assignment_count() const
{
    return m_copy_assignment_count;
}

std::size_t CopyMoveCounter::move_constructed_count() const
{
    return m_move_constructed_count;
}

std::size_t CopyMoveCounter::move_assignment_count() const
{
    return m_move_assignment_count;
}

std::size_t CopyMoveCounter::copy_count() const
{
    return m_copy_constructed_count + m_copy_assignment_count;
}

std::size_t CopyMoveCounter::move_count() const
{
    return m_move_constructed_count + m_move_assignment_count;
}

bool CopyMoveCounter::was_copied() const
{
    return m_was_copied;
}

bool CopyMoveCounter::was_moved() const
{
    return m_was_moved;
}

void CopyMoveCounter::inc()
{
    m_value++;
}

int CopyMoveCounter::value() const
{
    return m_value;
}

void CopyMoveCounter::reset()
{
    global_default_constructed_count = 0;
    global_value_constructed_count   = 0;
    global_copy_constructed_count    = 0;
    global_move_constructed_count    = 0;
    global_copy_assignment_count     = 0;
    global_move_assignment_count     = 0;
}

int CopyMoveCounter::global_move_count()
{
    return global_move_constructed_count + global_move_assignment_count;
}

int CopyMoveCounter::global_copy_count()
{
    return global_copy_constructed_count + global_copy_assignment_count;
}

}  // namespace mrc
