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

#include "srf/memory/adaptors.hpp"
#include "srf/utils/bytes_to_string.hpp"

#include <glog/logging.h>

namespace srf::memory {

template <typename Upstream>
class logging_resource final : public upstream_resource<Upstream>
{
  public:
    logging_resource(Upstream upstream, std::string prefix = "") :
      upstream_resource<Upstream>(std::move(upstream), "logger"),
      m_prefix(std::move(prefix))
    {}
    ~logging_resource() override = default;

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) final
    {
        auto ptr = this->resource()->allocate(bytes, alignment);
        LOG(INFO) << m_prefix << ": allocated " << ptr << "; size=" << bytes_to_string(bytes);
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) final
    {
        LOG(INFO) << m_prefix << ": deallocating " << ptr << "; size=" << bytes_to_string(bytes);
        this->resource()->deallocate(ptr, bytes, alignment);
    }

    std::string m_prefix;
};

}  // namespace srf::memory
