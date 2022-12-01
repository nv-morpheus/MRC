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

namespace mrc::codable {

class EncodingOptions final
{
  public:
    EncodingOptions() = default;
    EncodingOptions(const bool& force_copy, const bool& use_shm) : m_force_copy{force_copy}, m_use_shm{use_shm} {};

    const bool& force_copy() const
    {
        return m_force_copy;
    }

    void force_copy(const bool& flag)
    {
        m_force_copy = flag;
    }

    const bool& use_shm() const
    {
        return m_use_shm;
    }

    void use_shm(const bool& flag)
    {
        m_use_shm = flag;
    }

  private:
    bool m_use_shm{false};
    bool m_force_copy{false};
};

}  // namespace mrc::codable
