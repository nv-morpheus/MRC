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

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>

namespace mrc::utils {

/**
 * @brief thread local storage for a std::shared_ptr<ResourceT>
 *
 * @tparam ResourceT
 */
template <typename ResourceT>
class ThreadLocalSharedPointer final
{
  public:
    inline static std::shared_ptr<ResourceT> get()
    {
        if (thread_local_ref() == nullptr)
        {
            throw exceptions::MrcRuntimeError("accessing unset ThreadLocalSharedPointer");
        }
        return thread_local_ref();
    }

    // todo(ryan) - make private - friend to System and Runner
    inline static void set(std::shared_ptr<ResourceT> value)
    {
        thread_local_ref() = std::move(value);
    }

  protected:
    inline static std::shared_ptr<ResourceT>& thread_local_ref()
    {
        thread_local std::shared_ptr<ResourceT> instance{nullptr};
        return instance;
    }

  private:
    ThreadLocalSharedPointer()  = default;
    ~ThreadLocalSharedPointer() = default;

    DELETE_MOVEABILITY(ThreadLocalSharedPointer);
    DELETE_COPYABILITY(ThreadLocalSharedPointer);
};

}  // namespace mrc::utils
