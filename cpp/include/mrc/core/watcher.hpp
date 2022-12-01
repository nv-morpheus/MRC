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

#include <memory>
#include <set>

namespace mrc {

#ifdef MRC_TRACING_DISABLED
    #define WATCHER_PROLOGUE(event)
    #define WATCHER_EPILOGUE(event, rc)
    #define WATCHER_EPILOGUE(event, rc)
#else
    #define WATCHER_PROLOGUE(event) Watchable::watcher_prologue((event), this)
    #define WATCHER_EPILOGUE(event, rc) Watchable::watcher_epilogue((event), (rc), this)
    #define WATCHER_EPILOGUE(event, rc) Watchable::watcher_epilogue((event), (rc), this)
#endif

enum class WatchableEvent
{
    channel_write,
    channel_read,
    sink_on_data,
};

struct WatcherInterface
{
    virtual ~WatcherInterface() = default;

    virtual void on_entry(const WatchableEvent&, const void*)      = 0;
    virtual void on_exit(const WatchableEvent&, bool, const void*) = 0;
};

class Watchable
{
  public:
    void add_watcher(std::shared_ptr<WatcherInterface> /*obs*/);
    void remove_watcher(std::shared_ptr<WatcherInterface> /*obs*/);

  protected:
    inline void watcher_prologue(WatchableEvent /*op*/, const void* addr);
    inline void watcher_epilogue(WatchableEvent /*op*/, bool /*rc*/, const void* addr);

  private:
    std::set<std::shared_ptr<WatcherInterface>> m_watchers;
};

inline void Watchable::add_watcher(std::shared_ptr<WatcherInterface> obs)
{
    m_watchers.insert(obs);
}

inline void Watchable::remove_watcher(std::shared_ptr<WatcherInterface> obs)
{
    m_watchers.erase(obs);
}

inline void Watchable::watcher_prologue(WatchableEvent op, const void* addr)
{
    for (const auto& obs : m_watchers)
    {
        obs->on_entry(op, addr);
    }
}

inline void Watchable::watcher_epilogue(WatchableEvent op, bool rc, const void* addr)
{
    for (const auto& obs : m_watchers)
    {
        obs->on_exit(op, rc, addr);
    }
}

}  // namespace mrc
