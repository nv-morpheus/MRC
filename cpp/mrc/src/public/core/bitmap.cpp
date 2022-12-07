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

#include "mrc/core/bitmap.hpp"

#include "internal/utils/parse_ints.hpp"
#include "internal/utils/ranges.hpp"

#include <glog/logging.h>
#include <hwloc/bitmap.h>

#include <utility>  // for exchange

#define CHECK_HWLOC(hwloc_call) \
    {                           \
        auto rc = hwloc_call;   \
        CHECK_NE(rc, -1);       \
    }

namespace mrc {

Bitmap::Bitmap() : m_bitmap(hwloc_bitmap_alloc()) {}

Bitmap::Bitmap(int cpu_id) : m_bitmap(hwloc_bitmap_alloc())
{
    on(cpu_id);
}

Bitmap::Bitmap(const std::string& cpu_str) : m_bitmap(hwloc_bitmap_alloc())
{
    auto cpus = parse_ints(cpu_str);
    for (const auto& cpu : cpus)
    {
        on(cpu);
    }
}
Bitmap::Bitmap(hwloc_const_bitmap_t bitmap)
{
    m_bitmap = hwloc_bitmap_dup(bitmap);
}
Bitmap::Bitmap(const Bitmap& other)
{
    CHECK(other.m_bitmap);
    m_bitmap = hwloc_bitmap_dup(other.m_bitmap);
}
Bitmap& Bitmap::operator=(const Bitmap& other)
{
    CHECK(other.m_bitmap);
    m_bitmap = hwloc_bitmap_dup(other.m_bitmap);
    return *this;
}
Bitmap::~Bitmap()
{
    if (m_bitmap != nullptr)
    {
        hwloc_bitmap_free(m_bitmap);
    }
}
Bitmap::Bitmap(Bitmap&& other) noexcept : m_bitmap(std::exchange(other.m_bitmap, nullptr)) {}
Bitmap& Bitmap::operator=(Bitmap&& other) noexcept
{
    m_bitmap = std::exchange(other.m_bitmap, nullptr);
    return *this;
}
void Bitmap::on(std::uint32_t id)
{
    auto rc = hwloc_bitmap_set(m_bitmap, id);
    CHECK_NE(rc, -1);
}
void Bitmap::off(std::uint32_t id)
{
    CHECK_HWLOC(hwloc_bitmap_clr(m_bitmap, id));
}
void Bitmap::only(std::uint32_t id)
{
    auto rc = hwloc_bitmap_only(m_bitmap, id);
    CHECK_NE(rc, -1);
}
hwloc_bitmap_s& Bitmap::bitmap()
{
    CHECK(m_bitmap);
    return *m_bitmap;
}
const hwloc_bitmap_s& Bitmap::bitmap() const
{
    CHECK(m_bitmap);
    return *m_bitmap;
}
std::uint32_t Bitmap::first() const
{
    return next();
}
std::uint32_t Bitmap::next(int prev) const
{
    auto rc = hwloc_bitmap_next(m_bitmap, prev);
    return rc;
}
int Bitmap::weight() const
{
    auto rc = hwloc_bitmap_weight(m_bitmap);
    CHECK_NE(rc, -1);
    return rc;
}

bool Bitmap::is_set(int cpu_id) const
{
    CHECK_GE(cpu_id, 0);
    auto rc = hwloc_bitmap_isset(m_bitmap, cpu_id);
    return bool(rc);
}

bool Bitmap::empty() const
{
    return (hwloc_bitmap_iszero(m_bitmap) == 1);
}
Bitmap Bitmap::set_intersect(const Bitmap& other) const
{
    Bitmap result;
    CHECK_HWLOC(hwloc_bitmap_and(&result.bitmap(), m_bitmap, &other.bitmap()));
    return result;
}
Bitmap Bitmap::set_union(const Bitmap& other) const
{
    Bitmap result;
    CHECK_HWLOC(hwloc_bitmap_or(&result.bitmap(), m_bitmap, &other.bitmap()));
    return result;
}

void Bitmap::append(const Bitmap& bitmap)
{
    bitmap.for_each_bit([this](std::uint32_t idx, std::uint32_t bit) { this->on(bit); });
}

std::vector<std::uint32_t> Bitmap::vec() const
{
    std::uint32_t index;
    std::vector<std::uint32_t> v;
    hwloc_bitmap_foreach_begin(index, m_bitmap)
    {
        v.push_back(index);
    }
    hwloc_bitmap_foreach_end();
    return v;
}
std::string Bitmap::str() const
{
    return print_ranges(find_ranges(vec()));
}
void Bitmap::for_each_bit(std::function<void(std::uint32_t, std::uint32_t)> lambda) const
{
    const auto count = weight();
    for (int i = 0, prev = -1; i < count; i++)
    {
        prev = next(prev);
        lambda(i, prev);
    }
}
void Bitmap::zero()
{
    hwloc_bitmap_zero(m_bitmap);
}

Bitmap Bitmap::pop(std::size_t nbits)
{
    Bitmap rv;
    CHECK_LE(nbits, weight()) << "pop requesting more bits than set in bitmap";

    for (long i = 0, prev = -1; i < nbits; ++i)
    {
        prev = next(prev);
        DCHECK_NE(prev, -1);
        rv.on(prev);
        off(prev);
    }

    return rv;
}

std::vector<Bitmap> Bitmap::split(int nways) const
{
    CHECK_GT(nways, 0);
    int div = weight() / nways;
    int rem = weight() % nways;

    Bitmap copy = *this;
    std::vector<Bitmap> v;
    for (int i = 0; i < nways; ++i)
    {
        int count = div + (i < rem ? 1 : 0);
        if (count != 0)
        {
            v.push_back(copy.pop(count));
        }
        else
        {
            v.emplace_back();
        }
    }
    return v;
}

RoundRobinCpuSet::RoundRobinCpuSet(CpuSet bits) : m_bits(std::move(bits)), m_prev(-1), m_index(-1) {}
std::pair<int, int> RoundRobinCpuSet::next()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto id = hwloc_bitmap_next(&m_bits.bitmap(), m_prev);
    m_prev  = id;
    m_index++;
    if (id == -1)
    {
        m_index = -1;
        lock.unlock();
        return next();
    }
    return std::make_pair(m_index, id);
}
int RoundRobinCpuSet::next_index()
{
    auto pair = next();
    return pair.first;
}
int RoundRobinCpuSet::next_id()
{
    auto pair = next();
    return pair.second;
}
CpuSet RoundRobinCpuSet::next_binding()
{
    auto index = next_id();
    CpuSet cpu_set;
    cpu_set.only(index);
    return cpu_set;
}
void RoundRobinCpuSet::reset()
{
    m_prev = m_index = -1;
}
const CpuSet& RoundRobinCpuSet::cpu_set() const
{
    return m_bits;
}

bool Bitmap::contains(const Bitmap& sub_bitmap) const
{
    return bool(hwloc_bitmap_isincluded(&sub_bitmap.bitmap(), m_bitmap));
}
}  // namespace mrc
