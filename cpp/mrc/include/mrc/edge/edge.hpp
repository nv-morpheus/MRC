/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <sys/types.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc::edge {

class EdgeLifetime
{
  public:
    EdgeLifetime(std::function<void()> destruct_fn, bool is_armed = true) :
      m_destruct_fn(std::move(destruct_fn)),
      m_is_armed(is_armed)
    {}

    EdgeLifetime(const EdgeLifetime& other) = delete;
    EdgeLifetime(EdgeLifetime&& other)
    {
        std::swap(m_is_armed, other.m_is_armed);
        std::swap(m_destruct_fn, other.m_destruct_fn);
    }

    EdgeLifetime& operator=(const EdgeLifetime& other) = delete;
    EdgeLifetime& operator=(EdgeLifetime&& other) noexcept
    {
        std::swap(m_is_armed, other.m_is_armed);
        std::swap(m_destruct_fn, other.m_destruct_fn);

        return *this;
    }

    ~EdgeLifetime()
    {
        if (m_is_armed && m_destruct_fn)
        {
            m_destruct_fn();
        }
    }

    bool is_armed() const
    {
        return m_is_armed;
    }

    void arm()
    {
        m_is_armed = true;
    }

    void disarm()
    {
        m_is_armed = false;
    }

  private:
    bool m_is_armed{true};
    std::function<void()> m_destruct_fn;
};

class EdgeBase
{
  public:
    virtual ~EdgeBase()
    {
        for (auto& c : m_connectors)
        {
            c.disarm();
        }

        m_connectors.clear();
        m_disconnectors.clear();
    };

    bool is_connected() const
    {
        return m_is_connected;
    }

    void add_connector(std::function<void()>&& on_connect_fn)
    {
        this->add_connector(EdgeLifetime(std::move(on_connect_fn), true));
    }

    void add_connector(EdgeLifetime&& connector)
    {
        m_connectors.emplace_back(std::move(connector));
    }

    void add_disconnector(std::function<void()>&& on_disconnect_fn)
    {
        this->add_disconnector(EdgeLifetime(std::move(on_disconnect_fn), false));
    }

    void add_disconnector(EdgeLifetime&& disconnector)
    {
        LOG_IF(WARNING, disconnector.is_armed()) << "Adding armed disconnector to edge. This will fire even if edge is "
                                                    "never connected";

        m_disconnectors.emplace_back(std::move(disconnector));
    }

  protected:
    void connect()
    {
        m_is_connected = true;

        // Clear the connectors to execute them
        m_connectors.clear();

        // Arm all of the disconnectors now that we are connected
        for (auto& c : m_disconnectors)
        {
            c.arm();
        }

        // For all linked edges, call connect
        for (auto& linked_edge : m_linked_edges)
        {
            linked_edge->connect();
        }
    }

    void add_linked_edge(std::shared_ptr<EdgeBase> linked_edge)
    {
        if (m_is_connected)
        {
            linked_edge->connect();
        }

        m_linked_edges.emplace_back(std::move(linked_edge));
    }

  private:
    bool m_is_connected{false};
    std::vector<EdgeLifetime> m_connectors;
    std::vector<EdgeLifetime> m_disconnectors;
    std::vector<std::shared_ptr<EdgeBase>> m_linked_edges;

    // Friend any type of edge handle to allow calling connect
    template <typename>
    friend class Edge;
};

/**
 * @brief Typed version of `EdgeBase` that friends the holder classes
 *
 * @tparam T
 */
template <typename T>
class Edge : public virtual EdgeBase
{
  public:
    // Friend the holder classes which are required to setup connections
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

class EdgeTypeInfo
{
  public:
    EdgeTypeInfo(const EdgeTypeInfo& other) = default;

    std::type_index full_type() const
    {
        if (m_is_deferred)
        {
            throw std::runtime_error("Should not call full_type() for deferred types. Check is_deferred() first.");
        }

        return m_full_type.value();
    }

    std::type_index unwrapped_type() const
    {
        if (m_is_deferred)
        {
            throw std::runtime_error("Should not call unwrapped_type() for deferred types. Check is_deferred() first.");
        }

        return m_unwrapped_type.value();
    }

    bool is_deferred() const
    {
        return m_is_deferred;
    }

    bool operator==(const EdgeTypeInfo& other) const
    {
        return m_is_deferred == other.m_is_deferred && m_full_type == other.m_full_type &&
               m_unwrapped_type == other.m_unwrapped_type;
    }

    template <typename T>
    static EdgeTypeInfo create()
    {
        if constexpr (is_smart_ptr<T>::value)
        {
            return {typeid(T), typeid(typename T::element_type), false};
        }
        else
        {
            return {typeid(T), typeid(T), false};
        }
    }

    static EdgeTypeInfo create_deferred()
    {
        return {std::nullopt, std::nullopt, true};
    }

  private:
    EdgeTypeInfo(std::optional<std::type_index> full_type,
                 std::optional<std::type_index> unwrapped_type,
                 bool is_deferred) :
      m_full_type(full_type),
      m_unwrapped_type(unwrapped_type),
      m_is_deferred(is_deferred)
    {
        if (m_full_type.has_value())
        {
            m_full_type_str = type_name(m_full_type.value());
        }

        if (m_unwrapped_type.has_value())
        {
            m_unwrapped_type_str = type_name(m_unwrapped_type.value());
        }

        CHECK((m_is_deferred && !m_full_type.has_value() && !m_unwrapped_type.has_value()) ||
              (!m_is_deferred && m_full_type.has_value() && m_unwrapped_type.has_value()))
            << "Inconsistent deferred setting with concrete types";
    }

    std::optional<std::type_index> m_full_type;       // Includes any wrappers like shared_ptr
    std::string m_full_type_str;                      // For debugging purposes only
    std::optional<std::type_index> m_unwrapped_type;  // Excludes any wrappers like shared_ptr if they exist
    std::string m_unwrapped_type_str;                 // For debugging purposes only
    bool m_is_deferred{false};                        // Whether or not this type is deferred or concrete
};

class EdgeHandle
{
  public:
    const EdgeTypeInfo& get_type() const
    {
        return m_type;
    }

  protected:
    EdgeHandle(EdgeTypeInfo type_pair, std::shared_ptr<EdgeBase> edge_handle) :
      m_type(type_pair),
      m_handle(std::move(edge_handle))
    {}

    std::shared_ptr<EdgeBase> get_handle() const
    {
        return m_handle;
    }

    template <typename T>
    std::shared_ptr<T> get_handle_typed() const
    {
        return std::dynamic_pointer_cast<T>(m_handle);
    }

  private:
    EdgeTypeInfo m_type;

    std::shared_ptr<EdgeBase> m_handle{};

    // Allow ingress and egress derived objects to specialize
    friend WritableEdgeHandle;
    friend ReadableEdgeHandle;

    // Add EdgeHandle to unpack the object before discarding
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};
}  // namespace mrc::edge
