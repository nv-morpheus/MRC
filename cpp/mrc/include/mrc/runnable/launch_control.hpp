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

#include "mrc/constants.hpp"
#include "mrc/core/bitmap.hpp"
#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/fiber_pool.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/forward.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/engine_factory.hpp"
#include "mrc/runnable/internal_service.hpp"
#include "mrc/runnable/launch_control_config.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/runnable/runner_event.hpp"
#include "mrc/runnable/type_traits.hpp"
#include "mrc/runnable/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace mrc::runnable {

/**
 * @brief Coordinates the requires resources needed to launch a Runnable
 *
 * The steps required to launch a Runnable are:
 * - based on a set of LaunchOptions, build some number of Engines of the proper type
 * - construct a Context of the proper type for each instance of the Runnable
 * - create a Runner object which will manage the lifecycle of the Runnable
 * - create a Launcher which is a one-time use object to launch the Runnable and return the Runner
 *
 * This is too much too coordinate, too many object and lookups, to do at the location the Runnable is constructed, thus
 * the LaunchControl is object that brings all those pieces together and can provide a Launcher to the Runnable at the
 * place in the code it is constructured.
 *
 * LaunchControl is complext object to configure as it requires data from the primary Options passed to the Executor and
 * SegmentObject specific options from each Source/Sink/Node.
 *
 */
class LaunchControl final
{
  public:
    LaunchControl(LaunchControlConfig&& config) : m_config(std::move(config))
    {
        // ensure the default group exists - this will error if not
        get_engine_factory(default_engine_factory_name());
    }

    /**
     * @brief Construct a Launcher for a Runnable
     *
     * LaunchControl will attempt to discover LaunchOptions for the Runnable. If none are found, the default
     * LaunchOptions are used.
     *
     * @tparam RunnableT
     * @tparam ContextArgsT
     * @param runnable
     * @param context_args
     * @return std::unique_ptr<Launcher>
     */
    template <template <typename> typename ContextWrapperT, typename RunnableT, typename... ContextArgsT>
    [[nodiscard]] std::unique_ptr<Launcher> prepare_launcher_with_wrapped_context(const LaunchOptions& options,
                                                                                  std::unique_ptr<RunnableT> runnable,
                                                                                  ContextArgsT&&... context_args)
    {
        // inspect runnable to make the proper contexts
        CHECK(runnable) << "Null Runnable detected";
        using context_t = unwrap_context_t<runnable_context_t<RunnableT>>;

        VLOG(10) << "preparing engines using engine factory " << options.engine_factory_name
                 << "; pe_count=" << options.pe_count << "; engines_per_pe: " << options.engines_per_pe;

        // our launcher needs engines specific to the backend
        // engines are out way of running some task on the specified backend
        std::shared_ptr<Engines> engines = build_engines(options);

        // make contexts
        std::vector<std::shared_ptr<Context>> contexts;
        if constexpr (is_fiber_runnable_v<RunnableT>)
        {
            CHECK(get_engine_factory(options.engine_factory_name).backend() == EngineType::Fiber)
                << "Requested FiberRunnable to be run on a ThreadEngine";

            contexts = make_contexts<FiberContext<ContextWrapperT<context_t>>>(
                *engines, std::forward<ContextArgsT>(context_args)...);
        }
        else if constexpr (is_thread_context_v<RunnableT>)
        {
            CHECK(get_engine_factory(options.engine_factory_name).backend() == EngineType::Thread)
                << "Requested ThreadRunnable to be run on a FiberEngine";
            contexts = make_contexts<ThreadContext<ContextWrapperT<context_t>>>(
                *engines, std::forward<ContextArgsT>(context_args)...);
        }
        else
        {
            auto backend = get_engine_factory(options.engine_factory_name).backend();
            if (backend == EngineType::Fiber)
            {
                contexts = make_contexts<FiberContext<ContextWrapperT<context_t>>>(
                    *engines, std::forward<ContextArgsT>(context_args)...);
            }
            else if (backend == EngineType::Thread)
            {
                contexts = make_contexts<ThreadContext<ContextWrapperT<context_t>>>(
                    *engines, std::forward<ContextArgsT>(context_args)...);
            }
            else
            {
                LOG(FATAL) << "Unsupported EngineType";
            }
        }

        // create runner
        auto runner = runnable::make_runner(std::move(runnable));

        // construct the launcher
        return std::make_unique<Launcher>(std::move(runner), std::move(contexts), std::move(engines));
    }

    /**
     * @brief Construct a Launcher for a Runnable
     *
     * LaunchControl will attempt to discover LaunchOptions for the Runnable. If none are found, the default
     * LaunchOptions are used.
     *
     * @tparam RunnableT
     * @tparam ContextArgsT
     * @param runnable
     * @param context_args
     * @return std::unique_ptr<Launcher>
     */
    template <typename RunnableT, typename... ContextArgsT>
    [[nodiscard]] std::unique_ptr<Launcher> prepare_launcher(std::unique_ptr<RunnableT> runnable,
                                                             ContextArgsT&&... context_args)
    {
        LaunchOptions options;
        return prepare_launcher(options, std::move(runnable), std::forward<ContextArgsT>(context_args)...);
    }

    /**
     * @brief Construct a Launcher for a Runnable
     *
     * In this overload, the user provides the LaunchOptions explicitly. This method provides the
     * implemenation of the core function of LaunchControl.
     *
     * @tparam RunnableT
     * @tparam ContextArgsT
     * @param options
     * @param runnable
     * @param context_args
     * @return std::unique_ptr<Launcher>
     */
    // std::enable_if_t<not(is_fiber_runnable_v<RunnableT> and is_thread_runnable_v<RunnableT>)>>
    template <typename RunnableT, typename... ContextArgsT>
    [[nodiscard]] std::unique_ptr<Launcher> prepare_launcher(const LaunchOptions& options,
                                                             std::unique_ptr<RunnableT> runnable,
                                                             ContextArgsT&&... context_args)
    {
        // inspect runnable to make the proper contexts
        CHECK(runnable) << "Null Runnable detected";
        using context_t = runnable_context_t<RunnableT>;

        VLOG(10) << "preparing engines using engine factory " << options.engine_factory_name
                 << "; pe_count=" << options.pe_count << "; engines_per_pe: " << options.engines_per_pe;

        // our launcher needs engines specific to the backend
        // engines are out way of running some task on the specified backend
        std::shared_ptr<Engines> engines = build_engines(options);

        // make contexts
        std::vector<std::shared_ptr<Context>> contexts;
        if constexpr (is_fiber_runnable_v<RunnableT>)
        {
            CHECK(get_engine_factory(options.engine_factory_name).backend() == EngineType::Fiber)
                << "Requested FiberRunnable to be run on a ThreadEngine";
            contexts = make_contexts<context_t>(*engines, std::forward<ContextArgsT>(context_args)...);
        }
        else if constexpr (is_thread_context_v<RunnableT>)
        {
            CHECK(get_engine_factory(options.engine_factory_name).backend() == EngineType::Thread)
                << "Requested ThreadRunnable to be run on a FiberEngine";
            contexts = make_contexts<context_t>(*engines, std::forward<ContextArgsT>(context_args)...);
        }
        else
        {
            auto backend = get_engine_factory(options.engine_factory_name).backend();
            if (backend == EngineType::Fiber)
            {
                contexts =
                    make_contexts<FiberContext<context_t>>(*engines, std::forward<ContextArgsT>(context_args)...);
            }
            else if (backend == EngineType::Thread)
            {
                contexts =
                    make_contexts<ThreadContext<context_t>>(*engines, std::forward<ContextArgsT>(context_args)...);
            }
            else
            {
                LOG(FATAL) << "Unsupported EngineType";
            }
        }

        // create runner
        auto runner = runnable::make_runner(std::move(runnable));

        // construct the launcher
        return std::make_unique<Launcher>(std::move(runner), std::move(contexts), std::move(engines));
    }

    std::shared_ptr<core::FiberTaskQueue> main()
    {
        if (m_main)
        {
            return m_main;
        }

        // todo(ryan) - return main from system
        LOG(FATAL) << "fix me";
        return nullptr;
    }

  protected:
    /**
     * @brief Determine if the user has specified any specific options for the given Runnable
     *
     * @param runnable
     * @return LaunchOptions
     */
    // LaunchOptions get_options(Runnable const* const runnable)
    // {
    //     // is the runnable an internal service
    //     const auto* service_ptr = dynamic_cast<const InternalService*>(runnable);
    //     if (service_ptr != nullptr)
    //     {
    //         return config().services.service_options(service_ptr->service_type());
    //     }

    //     // is the runnable a segment object
    //     // implementation awaiting the integration of the new runnable with segment objects
    //     return config().default_options;
    // }

    std::shared_ptr<Engines> build_engines(const LaunchOptions& launch_options) const
    {
        return get_engine_factory(launch_options.engine_factory_name).build_engines(launch_options);
    }

    /**
     * @brief Get the resource group object
     *
     * @param name
     * @return EngineFactory&
     */
    EngineFactory& get_engine_factory(std::string name) const
    {
        auto search = config().resource_groups.find(name);
        CHECK(search != config().resource_groups.end()) << "unable to find launch group named: " << name;
        return *(search->second);
    }

    /**
     * @brief Get the resource group object casted an intended type.
     *
     * @tparam GroupT
     * @param name
     * @return GroupT&
     */
    template <typename GroupT>
    GroupT& get_engine_factory(std::string name)
    {
        auto& group     = get_engine_factory(name);
        auto* group_ptr = dynamic_cast<GroupT*>(&group);
        LOG_IF(FATAL, group_ptr == nullptr) << "Requested group << " << name << " is not GroupT";
        return *group_ptr;
    }

    /**
     * @brief Generate the specialized Contexts required, one for each instance of the Runnable to be launched.
     *
     * @tparam WrappedContextT
     * @tparam ArgsT
     * @param engines
     * @param args
     * @return auto
     */
    template <typename WrappedContextT, typename... ArgsT>
    auto make_contexts(const Engines& engines, ArgsT&&... args)
    {
        const auto size = engines.size();
        std::vector<std::shared_ptr<Context>> contexts;
        auto resources = std::make_shared<typename WrappedContextT::resource_t>(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            contexts.push_back(std::make_shared<WrappedContextT>(resources, i, size, args...));
        }
        return std::move(contexts);
    }

    /**
     * @brief Access the config
     *
     * @return const LaunchControlConfig&
     */
    const LaunchControlConfig& config() const
    {
        return m_config;
    }

  private:
    LaunchControlConfig m_config;
    std::shared_ptr<core::FiberTaskQueue> m_main;
};

}  // namespace mrc::runnable
