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

#include "mrc/runnable/context.hpp"
#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/fiber_context.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/runnable/thread_context.hpp"  // IWYU pragma: keep
#include "mrc/runnable/type_traits.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace mrc::runnable {

/**
 * @brief Runner takes ownership and manages the lifecycle of a Runnable
 *
 * Runner is an abstract base class that implements the majority of the Runner logic. Runner is an RAII object whose
 * lifecycle is tied to the owned Runnable. If the Runner goes out of scope before the Runnable is Completed, the
 * destructor will issue a kill and attempt to join all concurrent instances of the Runnable.
 *
 * make_runner is a convenience method designed to simplifiy the creation of a SpecializedRunner which publically
 * exposes the enqueue method that allows for the launch of the runnable on a specific execution backend.
 *
 * The Runner ownes both the Runnable and the Engine and provides access to them.
 *
 * A single callback can be added to the Runner which will be called immediate when the Runner's state is changed.
 * This callback is assume to be relatively lightweight. If there is a time component assocated with the callback, the
 * callback can push the state change data to a queue which can be processed by its own execution engine outside the
 * immediate scope of the callback method.
 *
 * After enqueued, the unique_ptr from make_runner maybe stored in any container that holds unique_ptr<Runnable>.
 */
class Runner
{
  protected:
    Runner(std::unique_ptr<Runnable>);

  public:
    enum class State
    {
        Unqueued = 0,
        Queued,
        Running,
        Error,
        Completed,
    };

    virtual ~Runner();

    DELETE_COPYABILITY(Runner);
    DELETE_MOVEABILITY(Runner);

    /**
     * @brief Signature for the callback lambda which is on each state change
     * Calling arguments are the new State and the unique instance id.
     */
    using on_instance_state_change_t = std::function<void(const Runnable&, std::size_t, State, State)>;

    /**
     * @brief Signature for the callback lambda which will be executed with bool value indicating the completion state
     * of the runnable.
     */
    using on_completion_callback_t = std::function<void(bool ok)>;

    /**
     * @brief Callback triggered on State change of individual Runnable context/instance
     *
     * @param callback
     */
    void on_instance_state_change_callback(on_instance_state_change_t callback);

    /**
     * @brief Completion callback triggered when a Runnable collectively completes.
     *
     * This callback is triggered only one time with a bool, where a true value indicated that all contexts/instances
     * finished without error. A false value indicates that one or more contexts/instances had uncaught exceptions.
     */
    void on_completion_callback(on_completion_callback_t callback);

    /**
     * @brief Fiber yielding call which returns when the Runnable is active
     */
    void await_live() const;

    /**
     * @brief Fiber yielding call which returns when the Runnable is complete
     */
    void await_join() const;

    /**
     * @brief Issues a request that the Runner terminates gracefully.
     */
    void stop() const;

    /**
     * @brief Issues a request that the Runner immediately terminates, ignoring any pending work
     */
    void kill() const;

    /**
     * @brief Access the const version of the Runnable
     */
    template <typename RunnableT>
    const RunnableT& runnable_as() const
    {
        CHECK(m_runnable);
        const RunnableT* ptr = dynamic_cast<RunnableT*>(m_runnable.get());
        CHECK(ptr != nullptr);
        return *ptr;
    }

    /**
     * @brief Object describing the state of an instance of a Runnable
     * Runnables can have multiple concurrent Instances. This object provides the details for one of these instances.
     */
    class Instance
    {
      public:
        std::size_t uid() const;
        State state() const;
        SharedFuture<void> live_future() const;
        SharedFuture<void> join_future() const;

      private:
        std::size_t m_uid{0};
        State m_state{State::Unqueued};
        Promise<void> m_live_promise;
        SharedFuture<void> m_live_future;
        SharedFuture<void> m_join_future;
        std::shared_ptr<Engine> m_engine;
        std::shared_ptr<Context> m_context;

        friend class Runner;
    };

    /**
     * @brief State of running instances
     * @return const std::vector<Instance>
     */
    const std::vector<Instance>& instances() const;

  protected:
    void enqueue(std::shared_ptr<Engines>, std::vector<std::shared_ptr<Context>>&&);

    /**
     * @brief Access the const version of the Runnable
     */
    Runnable& runnable();

  private:
    /**
     * @brief Advance the State of the Runner
     * @param new_state
     */
    void update_state(std::size_t launcher_id, State new_state);

    // callback lambda executed on state change
    on_instance_state_change_t m_on_instance_state_change{nullptr};

    // callback lambda executed on completion
    on_completion_callback_t m_completion_callback{nullptr};

    std::atomic<bool> m_status{true};
    std::atomic<std::size_t> m_remaining_instances{0};

    // the runnable owned by the runner
    // using shared_ptr to allow python access; otherwise a unique_ptr woudld used
    std::unique_ptr<Runnable> m_runnable;

    // 1:1 mapping to contexts, but hold the runner specific states for each instance
    mutable std::vector<Instance> m_instances;

    // simple bool to disable launching this runner/runnable
    bool m_can_run{true};

    mutable std::recursive_mutex m_mutex;

    friend class Launcher;
};

/**
 * @brief SpecializedRunner which exposes the ablilty to enqueue the Runnable on a specific execution backend
 *
 * @tparam ContextT
 */
template <typename ContextT>
class SpecializedRunner : public Runner
{
  public:
    SpecializedRunner(std::unique_ptr<RunnableWithContext<ContextT>> runnable) : Runner(std::move(runnable)) {}
    ~SpecializedRunner() override = default;

    template <typename... ArgsT>
    void enqueue(std::shared_ptr<Engines> launcher, ArgsT&&... args)
    {
        DCHECK(launcher && launcher->size());

        std::vector<std::shared_ptr<Context>> contexts;
        if (launcher->engine_type() == EngineType::Fiber)
        {
            using ctx_t = std::conditional_t<is_fiber_context_v<ContextT>, ContextT, FiberContext<ContextT>>;
            contexts    = make_contexts<ctx_t>(*launcher, std::forward<ArgsT>(args)...);
        }
        else if (launcher->engine_type() == EngineType::Thread)
        {
            using ctx_t = std::conditional_t<is_fiber_context_v<ContextT>, ContextT, FiberContext<ContextT>>;
            contexts    = make_contexts<ctx_t>(*launcher, std::forward<ArgsT>(args)...);
        }
        return Runner::enqueue(launcher, std::move(contexts));
    }

  protected:
    template <typename WrappedContextT, typename... ArgsT>
    auto make_contexts(const Engines& launcher, ArgsT&&... args)
    {
        const auto size = launcher.size();
        std::vector<std::shared_ptr<Context>> contexts;
        auto resources = std::make_shared<typename WrappedContextT::resource_t>(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            contexts.push_back(std::make_shared<WrappedContextT>(resources, i, size, std::forward<ArgsT>(args)...));
        }
        return std::move(contexts);
    }
};

template <typename RunnableT>
auto make_runner(std::unique_ptr<RunnableT> runnable)
{
    CHECK(runnable);
    using context_t = runnable_context_t<RunnableT>;
    return std::make_unique<SpecializedRunner<context_t>>(std::move(runnable));
}

}  // namespace mrc::runnable
