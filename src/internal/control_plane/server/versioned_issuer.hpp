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

#include "internal/control_plane/server/update_issuer.hpp"

#include "srf/protos/architect.pb.h"
#include "srf/utils/macros.hpp"

#include <cstdint>

namespace srf::internal::control_plane::server {

class VersionedIssuer : public UpdateIssuer
{
  public:
    VersionedIssuer() = default;

    DELETE_MOVEABILITY(VersionedIssuer);
    DELETE_COPYABILITY(VersionedIssuer);

    void issue_update() final
    {
        if (m_issued_nonce < m_current_nonce)
        {
            m_issued_nonce = m_current_nonce;
            auto update    = make_update();
            do_issue_update(update);
        }
    }

  protected:
    void mark_as_modified()
    {
        m_current_nonce++;
    }

    protos::ServiceUpdate make_update() const
    {
        protos::ServiceUpdate update;
        update.set_service_name(this->service_name());
        update.set_nonce(m_current_nonce);
        do_make_update(update);
        return update;
    }

  private:
    virtual void do_make_update(protos::ServiceUpdate& update) const  = 0;
    virtual void do_issue_update(const protos::ServiceUpdate& update) = 0;

    std::size_t m_current_nonce{1};
    std::size_t m_issued_nonce{1};
};

}  // namespace srf::internal::control_plane::server
