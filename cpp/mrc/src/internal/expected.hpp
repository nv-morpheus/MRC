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

#include "mrc/utils/macros.hpp"
#include "mrc/utils/string_utils.hpp"  // IWYU pragma: export

#include <tl/expected.hpp>  // IWYU pragma: export

namespace mrc::internal {

enum class ErrorCode
{
    Internal,
    Fatal,
};

class Error;

using UnexpectedError = tl::unexpected<Error>;  // NOLINT

class Error final : public std::exception
{
    Error(ErrorCode type) : m_code(type) {}
    Error(std::string message) : Error(ErrorCode::Internal, std::move(message)) {}
    Error(ErrorCode type, std::string message) : m_code(type), m_message(std::move(message)) {}

  public:
    template <typename... ArgsT>
    static UnexpectedError create(ArgsT&&... args)
    {
        return UnexpectedError(Error(std::forward<ArgsT>(args)...));
    }

    DEFAULT_MOVEABILITY(Error);
    DEFAULT_COPYABILITY(Error);

    ErrorCode code() const
    {
        return m_code;
    }
    const std::string& message() const
    {
        return m_message;
    }

    const char* what() const noexcept final
    {
        return m_message.c_str();
    }

  private:
    ErrorCode m_code;
    std::string m_message;
};

template <typename T = void>
using Expected = tl::expected<T, Error>;  // NOLINT

#define MRC_CHECK(condition)                                                  \
    if (!(condition))                                                         \
    {                                                                         \
        return Error::create(MRC_CONCAT_STR("CHECK failed: " #condition "")); \
    }

#define MRC_EXPECT(expected)                                          \
    if (!(expected))                                                  \
    {                                                                 \
        DVLOG(10) << "expect failed: " << expected.error().message(); \
        return Error::create(std::move(expected.error()));            \
    }

#define MRC_THROW_ON_ERROR(expected)                                  \
    if (!(expected))                                                  \
    {                                                                 \
        DVLOG(10) << "expect failed: " << expected.error().message(); \
        throw expected.error();                                       \
    }

}  // namespace mrc::internal
