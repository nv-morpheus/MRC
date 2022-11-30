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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include "hwloc/bitmap.h"

struct hwloc_bitmap_s;
typedef struct hwloc_bitmap_s* hwloc_bitmap_t;              // NOLINT
typedef const struct hwloc_bitmap_s* hwloc_const_bitmap_t;  // NOLINT

namespace mrc {

class Bitmap
{
  public:
    Bitmap();
    Bitmap(int);
    Bitmap(const std::string&);
    Bitmap(hwloc_const_bitmap_t bitmap);
    virtual ~Bitmap();

    Bitmap(const Bitmap& other);
    Bitmap& operator=(const Bitmap& other);

    Bitmap(Bitmap&& other) noexcept;
    Bitmap& operator=(Bitmap&& other) noexcept;

    /**
     * @brief enables the bit for offset id
     **/
    void on(std::uint32_t id);

    /**
     * @brief disables the bit for offset id
     **/
    void off(std::uint32_t id);

    /**
     * @brief empty the bitmap and add (on) bit at offset id
     * @param [in] id
     **/
    void only(std::uint32_t id);

    /**
     * @brief clear / empty the bitmap
     **/
    void zero();

    /**
     * @brief index of the first on bit
     **/
    [[nodiscard]] std::uint32_t first() const;

    /**
     * @brief index of the next on bit after prev
     *
     * if `prev == -1`, then the first bit is returned
     **/
    [[nodiscard]] std::uint32_t next(int prev = -1) const;

    /**
     * @brief returns true of the bitmap is empty
     **/
    [[nodiscard]] bool empty() const;

    /**
     * @brief in place union of current bitmap with the incoming bitmap
     */
    void append(const Bitmap& bitmap);

    /**
     * @brief returns the set union of the current bit map the one passed in
     **/
    [[nodiscard]] Bitmap set_union(const Bitmap&) const;

    /**
     * @brief returns the set intersection of the current bit map the one passed in
     **/
    [[nodiscard]] Bitmap set_intersect(const Bitmap&) const;

    /**
     * @brief the number of indexes that are set on
     **/
    [[nodiscard]] int weight() const;

    /**
     * @brief determines if the bit at cpu_id is set (true) or not (false)
     *
     */
    [[nodiscard]] bool is_set(int cpu_id) const;

    /**
     * @brief Test whether bitmap sub_bitmap is completely contained as part of the current bitmap
     */
    bool contains(const Bitmap& sub_bitmap) const;

    /**
     * @brief returns a vector of set indexes
     **/
    [[nodiscard]] std::vector<std::uint32_t> vec() const;

    /**
     * @brief returns a string representation of the on bits, e.g. 0-7,16-23
     **/
    [[nodiscard]] std::string str() const;

    /**
     * @brief execute a lambda with the enumerated ith bit and ith index value as args
     **/
    void for_each_bit(std::function<void(std::uint32_t, std::uint32_t)>) const;

    /**
     * @brief Pop first N bits from the current bitmap and transfer them to the returned bitmap
     */
    Bitmap pop(std::size_t nbits);

    /**
     * @brief Splits the assigned bits in the bitmap the requested number of groups
     *
     * if (weight() % nways = remainder), then the first `remainder` bitmaps in the vector will have one more assigned
     * bits than the rest of the bitmaps in the vector.
     */
    std::vector<Bitmap> split(int nways) const;

    /**
     * @brief stream friendly representation
     **/
    friend std::ostream& operator<<(std::ostream& os, const Bitmap& bitmap)
    {
        os << "[bitmap - count: " << bitmap.weight() << "; str: " << bitmap.str() << "]";
        return os;
    }

    /**
     * @brief access the underlying hwloc bitmap
     */
    hwloc_bitmap_s& bitmap();

    /**
     * @brief access the underlying hwloc bitmap
     **/
    [[nodiscard]] const hwloc_bitmap_s& bitmap() const;

  private:
    hwloc_bitmap_t m_bitmap;
};

struct CpuSet : public Bitmap
{
    using Bitmap::Bitmap;

    CpuSet(Bitmap bitmap) : Bitmap(bitmap) {}

    friend std::ostream& operator<<(std::ostream& os, const CpuSet& bitmap)
    {
        os << "[cpu_set - count: " << bitmap.weight() << "; str: " << bitmap.str() << "]";
        return os;
    }
};

struct NumaSet : public Bitmap
{
    using Bitmap::Bitmap;
    friend std::ostream& operator<<(std::ostream& os, const NumaSet& bitmap)
    {
        os << "[numaset - count: " << bitmap.weight() << "; str: " << bitmap.str() << "]";
        return os;
    }
};

class RoundRobinCpuSet
{
  public:
    explicit RoundRobinCpuSet(CpuSet bits);
    virtual ~RoundRobinCpuSet() = default;

    /**
     * @brief returns the next enumerated pair (index, id) of the set
     **/
    std::pair<int, int> next();

    /**
     * @brief returns the next index where index is in [0, N) of the set size
     **/
    int next_index();

    /**
     * @brief returns the next id where id is the system-wide logical id of the object in the set
     **/
    int next_id();

    /**
     * @brief returns a cpu_set configured with the value from next_id
     **/
    CpuSet next_binding();

    /**
     * @brief reset the round robin cyclic pointer to the origin
     **/
    void reset();

    /**
     * @brief set of bits being round-robined by the provider
     **/
    [[nodiscard]] const CpuSet& cpu_set() const;

  private:
    CpuSet m_bits;
    int m_prev;
    int m_index;
    std::mutex m_mutex;
};

}  // namespace mrc
