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

namespace mrc {

class FiberPoolOptions
{
  public:
    FiberPoolOptions() = default;

    /**
     * @brief enable memory binding
     **/
    FiberPoolOptions& enable_memory_binding(bool default_true);

    /**
     * @brief enable thread binding
     **/
    FiberPoolOptions& enable_thread_binding(bool default_true);

    /**
     * @brief enable tracing scheduler
     **/
    FiberPoolOptions& enable_tracing_scheduler(bool default_false);

    [[nodiscard]] bool enable_memory_binding() const;
    [[nodiscard]] bool enable_thread_binding() const;
    [[nodiscard]] bool enable_tracing_scheduler() const;

  private:
    bool m_enable_memory_binding{true};
    bool m_enable_thread_binding{true};
    bool m_enable_tracing_scheduler{false};
};

}  // namespace mrc
