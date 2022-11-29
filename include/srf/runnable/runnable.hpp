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

#include "srf/runnable/context.hpp"
#include "srf/utils/macros.hpp"

#include <atomic>
#include <string>

namespace srf::runnable {

/**
 * Runnable is an abstract base class from which the deriving parent should override the pure virtual run() method
 * to define the business logic which will be run on the execution context provided by a Launcher.
 *
 * A Runner will take ownership of a Runnable via move, when this happens the State is transformed from Init -> Owned.
 * The public API of a Runnable should only be modified while it is in the Init state. It is upto the derived Runnables
 * to enforce this on their public API methods.
 */
class Runnable
{
    /**
     * @brief Override to provide the business logic for the Runnable
     */
    virtual void main(Context&) = 0;

  public:
    Runnable();
    virtual ~Runnable();

    DELETE_COPYABILITY(Runnable);
    DELETE_MOVEABILITY(Runnable);

  protected:
    enum class State
    {
        Init = 0,
        Owned,
        Run,
        Stop,
        Kill
    };

    /**
     * @brief Used to query the Runnable::State from derived classes
     * @return State
     */
    State state() const;

    /**
     * @brief Logging friendly string to distinguish the Runnable/Context pair
     * @return std::string
     */
    std::string info(const Context&) const;

  private:
    /**
     * @brief Updates the state, enforces forward only and calls on_state_change for valid updates
     *
     * This method will only be called from Runner and access will be protected by the Runner's internal mutex.
     */
    void update_state(State);

    /**
     * @brief Hook to perform actions on state updates
     */
    virtual void on_state_update(const State&);

    std::atomic<State> m_state{State::Init};

    friend class Runner;
};

template <typename ContextT>
class RunnableWithContext : public Runnable
{
  protected:
    using ContextType = ContextT;  // NOLINT

  private:
    virtual void run(ContextType&) = 0;

    void main(Context& context) final
    {
        run(context.as<ContextType>());
    }
};

}  // namespace srf::runnable
