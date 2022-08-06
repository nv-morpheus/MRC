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

#include "internal/system/engine_factory_cpu_sets.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/options/engine_groups.hpp"
#include "srf/options/options.hpp"
#include "srf/runnable/types.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <utility>

namespace srf::internal::system {

bool EngineFactoryCpuSets::is_resuable(const std::string& name) const
{
    auto search = reusable.find(name);
    CHECK(search != reusable.end()) << "unable to find " << name << " in reusable map";
    return search->second;
}

EngineFactoryCpuSets generate_engine_factory_cpu_sets(const Options& options, const CpuSet& cpu_set)
{
    EngineFactoryCpuSets config;

    auto cpu_count = cpu_set.weight();
    CHECK_GT(cpu_count, 0);

    const auto& services      = options.services();
    const auto& engine_groups = options.engine_factories();

    DVLOG(10) << "computing cpu_sets for engine factories";
    DVLOG(10) << "- using dedicated_main_thread: " << (engine_groups.dedicated_main_thread() ? "TRUE" : "FALSE");
    DVLOG(10) << "- default engine type        : " << runnable::engine_type_string(engine_groups.default_engine_type());

    // mutable map
    auto engine_groups_map = options.engine_factories().map();

    const bool specialized_main = options.engine_factories().dedicated_main_thread();
    const bool specialized_network =
        !options.architect_url().empty() && options.engine_factories().dedicated_network_thread();

    if (specialized_main)
    {
        DVLOG(10) << "main fiber is specialized; adding entry to engine_group_map";

        EngineFactoryOptions main;
        main.engine_type          = runnable::EngineType::Fiber;
        main.cpu_count            = 1;
        main.allow_overlap        = false;
        main.reusable             = true;
        engine_groups_map["main"] = std::move(main);
    }

    if (specialized_network)
    {
        EngineFactoryOptions net;
        net.engine_type                  = runnable::EngineType::Fiber;
        net.cpu_count                    = 1;
        net.allow_overlap                = false;
        net.reusable                     = true;
        engine_groups_map["srf_network"] = std::move(net);
    }

    DVLOG(10) << "evaluating minimum cpu count for engine group options";

    std::map<runnable::EngineType, std::size_t> engine_cpu_count;
    std::size_t shared_cpu_count = 0;

    for (const auto& kv : engine_groups_map)
    {
        DVLOG(10) << "- engine group `" << kv.first << "` requires " << kv.second.cpu_count << " logical cpus";
        if (kv.second.allow_overlap)
        {
            shared_cpu_count = std::max(shared_cpu_count, kv.second.cpu_count);

            if (kv.second.engine_type == runnable::EngineType::Fiber)
            {
                config.shared_cpus_has_fibers = true;
            }
        }
        else
        {
            engine_cpu_count[kv.second.engine_type] += kv.second.cpu_count;
        }
    }

    auto min_cpu_count = shared_cpu_count;
    for (const auto& kv : engine_cpu_count)
    {
        min_cpu_count += kv.second;
    }

    DVLOG(10) << "- engine group `main` requires 1 logical cpus";

    DVLOG(10) << "required logical cpus";
    DVLOG(10) << "- dedicated : " << min_cpu_count - shared_cpu_count;
    DVLOG(10) << "- shared    : " << shared_cpu_count;
    DVLOG(10) << "- total     : " << min_cpu_count + 1 << " (including main)";

    if (min_cpu_count + 1 /* for main */ > cpu_count)
    {
        LOG(ERROR) << "requested configuration requires " << min_cpu_count << " logical cpus; only " << cpu_count
                   << " detected";
        throw exceptions::SrfRuntimeError("insufficient number of logical cpus assigned to the current process");
    }

    // for the set of logical cpus in the placement group, first assign all cpus that will be in the fiber pool for this
    // group, starting with the default pool
    // the remaining logical cpus will be reserved for thread pools or thread runnables

    DVLOG(10) << "allocating logical cpus for `" << default_engine_factory_name() << "`` pool";
    auto remaining_cpu_set             = cpu_set;
    std::size_t default_pool_cpu_count = cpu_count - min_cpu_count;
    auto default_pool_cpu_set          = remaining_cpu_set.pop(default_pool_cpu_count);
    if (options.engine_factories().default_engine_type() == runnable::EngineType::Fiber)
    {
        config.fiber_cpu_sets[default_engine_factory_name()] = default_pool_cpu_set;
    }
    else if (options.engine_factories().default_engine_type() == runnable::EngineType::Thread)
    {
        config.thread_cpu_sets[default_engine_factory_name()] = default_pool_cpu_set;
    }
    else
    {
        LOG(FATAL) << "invalid default engine type: "
                   << runnable::engine_type_string(options.engine_factories().default_engine_type());
    }
    DVLOG(10) << "- cpu_set for `" << default_engine_factory_name() << "`` pool: " << default_pool_cpu_set;

    if (!specialized_main)
    {
        // if we are not using a dedicated main thread, main is allocated from the default pool
        DVLOG(10) << "allocating logical cpu for `main` - from the `" << default_engine_factory_name() << "` pool";
        CpuSet main_pool_cpu_set;
        main_pool_cpu_set.on(default_pool_cpu_set.first());
        config.fiber_cpu_sets["main"] = main_pool_cpu_set;
        DVLOG(10) << "- cpu_set for `main`: " << main_pool_cpu_set;
    }

    config.reusable[default_engine_factory_name()] = true;
    config.reusable["main"]                        = true;

    // if we are not using a dedicated network thread, use the same fiber queue as main for srf_network
    if (!options.architect_url().empty() && !specialized_network)
    {
        config.fiber_cpu_sets["srf_network"] = config.fiber_cpu_sets.at("main");
        config.reusable["srf_network"]       = true;
        DVLOG(10) << "- cpu_set for `srf_network`: " << config.fiber_cpu_sets["srf_network"];
    }

    // get all resources for groups that have overlap disabled
    DVLOG(10) << "allocating logical cpus for non-overlapping pools";
    for (const auto& kv : engine_groups_map)
    {
        config.reusable[kv.first] = kv.second.reusable;

        if (!kv.second.allow_overlap)
        {
            auto this_set = remaining_cpu_set.pop(kv.second.cpu_count);
            DVLOG(10) << "- cpu_set for non-overlapping `" << kv.first << "` pool: " << this_set;
            if (kv.second.engine_type == runnable::EngineType::Fiber)
            {
                config.fiber_cpu_sets[kv.first] = this_set;
            }
            else if (kv.second.engine_type == runnable::EngineType::Thread)
            {
                config.thread_cpu_sets[kv.first] = this_set;
            }
        }
    }

    config.shared_cpus_set = remaining_cpu_set;
    int idx                = -1;

    DVLOG(10) << "allocating logical cpus for overlapping pools using round robin distribution of the shared cpu set";
    for (const auto& kv : engine_groups_map)
    {
        if (kv.second.allow_overlap)
        {
            CpuSet this_set;

            for (int i = 0; i < kv.second.cpu_count; ++i)
            {
                // allow round robin distribution
                do
                {
                    idx = remaining_cpu_set.next(idx);
                } while (idx == -1);

                this_set.on(idx);
            }

            DVLOG(10) << "- cpu_set for overlapping `" << kv.first << "` pool: " << this_set;

            if (kv.second.engine_type == runnable::EngineType::Fiber)
            {
                config.fiber_cpu_sets[kv.first] = this_set;
            }
            else if (kv.second.engine_type == runnable::EngineType::Thread)
            {
                config.thread_cpu_sets[kv.first] = this_set;
            }
        }
    }

    return config;
}

std::size_t EngineFactoryCpuSets::main_cpu_id() const
{
    auto search = fiber_cpu_sets.find("main");
    CHECK(search != fiber_cpu_sets.end()) << "unable to lookup cpuset for main";
    CHECK_EQ(search->second.weight(), 1);
    return search->second.first();
}

}  // namespace srf::internal::system
