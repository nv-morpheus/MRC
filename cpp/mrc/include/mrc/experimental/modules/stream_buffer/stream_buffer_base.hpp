/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rxcpp/rx.hpp>

#include <cstddef>
#include <type_traits>

namespace mrc::modules::stream_buffers {

template <typename DataTypeT>
class StreamBufferBase
{
  public:
    virtual ~StreamBufferBase() = default;

    virtual std::size_t buffer_size() = 0;

    virtual void buffer_size(std::size_t size) = 0;

    virtual bool empty() = 0;

    virtual void push_back(DataTypeT&& data) = 0;

    virtual void flush_next(rxcpp::subscriber<DataTypeT>& subscriber) = 0;

    virtual void flush_all(rxcpp::subscriber<DataTypeT>& subscriber) = 0;
};
}  // namespace mrc::modules::stream_buffers
