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

#include "srf/utils/macros.hpp"

#include <boost/fiber/future/promise.hpp>
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>

#include <atomic>

namespace srf::internal::data_plane {

class Callbacks;
class Client;

class Request final
{
  public:
    Request();
    ~Request();

    DELETE_COPYABILITY(Request);
    DELETE_MOVEABILITY(Request);

    // std::optional<Status> is_complete();
    bool await_complete();

    // attempts to cancel the request
    // the request will either be cancelled or completed
    // void try_cancel();

  private:
    void reset();

    enum class State
    {
        Init,
        Running,
        OK,
        Cancelled,
        Error
    };
    std::atomic<State> m_state{State::Init};
    void* m_request{nullptr};
    void* m_rkey{nullptr};

    friend Client;
    friend Callbacks;
};

}  // namespace srf::internal::data_plane
