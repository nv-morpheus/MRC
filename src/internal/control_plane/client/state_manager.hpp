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

#include "mrc/node/source_channel.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace mrc::internal::control_plane {

class Client;

namespace client {

/**
 * todo(ryan) - we can probably remove this class as only ConnectionManager inherits from it
 * @note - it would be nice if we could improve the consensus model to enable more synchronous behavior to enable better
 * testing scenarios. for how, the update_future on this class is too valuable and it allows us to wait until all KNOWN
 * clients have connected and exchanged information
 */

class StateManager
{
  public:
    StateManager(Client& client);
    virtual ~StateManager();

    DELETE_COPYABILITY(StateManager);
    DELETE_MOVEABILITY(StateManager);

    /**
     * @brief Get a Future<void> which will be completed on the next update
     *
     * @note This future will only complete when an update is received for the specific state being requested or the
     * State is marked for removal. In the latter case where the State is marked for removal, an exception_ptr will be
     * set on the promise resulting in an excpetion being thrown on the get of the future.
     *
     * @return Future<void>
     */
    Future<void> update_future();

  protected:
    const Client& client() const;
    Client& client();

    void start_with_channel(node::SourceChannel<const protos::StateUpdate>& update_channel);
    void await_join();

  private:
    /**
     * @brief Triggers a do_update if the StateUpdate is more recent then the current state
     *
     * If and only if the current state is updated will the set of awaiting promises be completed.
     *
     * @param update_msg
     */
    void update(const protos::StateUpdate&& update_msg);

    virtual void do_update(const protos::StateUpdate&& update_msg) = 0;

    Client& m_client;
    std::size_t m_nonce{1};
    mutable std::mutex m_mutex;
    std::vector<Promise<void>> m_update_promises;
    std::unique_ptr<mrc::runnable::Runner> m_runner;
};

}  // namespace client
}  // namespace mrc::internal::control_plane
