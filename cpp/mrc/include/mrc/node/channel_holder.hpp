#pragma once

#include "mrc/channel/channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/string_utils.hpp"

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

namespace mrc::node {

struct virtual_enable_shared_from_this_base  // NOLINT(readability-identifier-naming)
  : public std::enable_shared_from_this<virtual_enable_shared_from_this_base>
{
    virtual ~virtual_enable_shared_from_this_base() = default;
};

// This class allows enable_shared_from_this to work for virtual inheritance
template <typename T>
struct virtual_enable_shared_from_this  // NOLINT(readability-identifier-naming)
  : public virtual virtual_enable_shared_from_this_base
{
    std::shared_ptr<T> shared_from_this()
    {
        return std::shared_ptr<T>(virtual_enable_shared_from_this_base::shared_from_this(), static_cast<T*>(this));
    }
    std::shared_ptr<const T> shared_from_this() const
    {
        return std::shared_ptr<const T>(virtual_enable_shared_from_this_base::shared_from_this(),
                                        static_cast<const T*>(this));
    }
};

// struct virtual_enable_shared_from_this_base : public
// std::enable_shared_from_this<virtual_enable_shared_from_this_base>
// {
//     virtual ~virtual_enable_shared_from_this_base() {}
// };
// template <typename T>
// struct virtual_enable_shared_from_this : public virtual virtual_enable_shared_from_this_base
// {
//     std::shared_ptr<T> shared_from_this()
//     {
//         return std::dynamic_pointer_cast<T>(virtual_enable_shared_from_this_base::shared_from_this());
//     }
// };

template <typename T>
class EdgeHolder;

template <typename KeyT, typename T>
class MultiEdgeHolder;

template <typename T>
class EdgeHandle;

struct IngressHandleObj;
struct EgressHandleObj;

class EdgeLifetime
{
  public:
    EdgeLifetime(std::function<void()> fn) : m_fn(std::move(fn)) {}

    EdgeLifetime(const EdgeLifetime& other) = delete;
    EdgeLifetime(EdgeLifetime&& other)
    {
        std::swap(m_fn, other.m_fn);
    }

    EdgeLifetime& operator=(const EdgeLifetime& other) = delete;
    EdgeLifetime& operator=(EdgeLifetime&& other) noexcept
    {
        std::swap(m_fn, other.m_fn);

        return *this;
    }

    ~EdgeLifetime()
    {
        if (m_fn)
        {
            m_fn();
        }
    }

    void disarm()
    {
        m_fn = nullptr;
    }

  private:
    std::function<void()> m_fn;
};

class EdgeTag
{
  public:
    virtual ~EdgeTag()
    {
        VLOG(10) << "Destroying EdgeHandle";

        for (auto& c : m_connectors)
        {
            c.disarm();
        }

        m_disconnectors.clear();
    };

    bool is_connected() const
    {
        return m_is_connected;
    }

    void add_connector(EdgeLifetime&& connector)
    {
        m_connectors.emplace_back(std::move(connector));
    }

    void add_disconnector(EdgeLifetime&& disconnector)
    {
        m_disconnectors.emplace_back(std::move(disconnector));
    }

  protected:
    void connect()
    {
        m_is_connected = true;

        // Clear the connectors to execute them
        m_connectors.clear();

        // For all linked edges, call connect
        for (auto& linked_edge : m_linked_edges)
        {
            linked_edge->connect();
        }
    }

    void add_linked_edge(std::shared_ptr<EdgeTag> linked_edge)
    {
        if (m_is_connected)
        {
            linked_edge->connect();
        }

        m_linked_edges.emplace_back(std::move(linked_edge));
    }

    // // Allows keeping a downstream edge holder alive for the lifetime of this edge
    // void add_lifetime(std::shared_ptr<EdgeHolder<T>> downstream)
    // {
    //     m_keep_alive.push_back(downstream);
    // }

  private:
    bool m_is_connected{false};
    std::vector<EdgeLifetime> m_connectors;
    std::vector<EdgeLifetime> m_disconnectors;
    std::vector<std::shared_ptr<EdgeTag>> m_linked_edges;

    // Friend any type of edge handle to allow calling connect
    template <typename>
    friend class EdgeHandle;
};

template <typename T>
class EdgeHandle : public virtual EdgeTag
{
  public:
    // ~EdgeHandle() override
    // {
    //     VLOG(10) << "Destroying EdgeHandle";

    //     for (auto& c : m_connectors)
    //     {
    //         c.disarm();
    //     }

    //     m_disconnectors.clear();
    // };

    // bool is_connected() const override
    // {
    //     return m_is_connected;
    // }

    // void add_connector(EdgeLifetime&& connector)
    // {
    //     m_connectors.emplace_back(std::move(connector));
    // }

    // void add_disconnector(EdgeLifetime&& disconnector)
    // {
    //     m_disconnectors.emplace_back(std::move(disconnector));
    // }

    //   protected:
    //     void connect() override
    //     {
    //         m_is_connected = true;

    //         // Clear the connectors to execute them
    //         m_connectors.clear();

    //         // For all linked edges, call connect
    //         for (auto& linked_edge : m_linked_edges)
    //         {
    //             linked_edge->connect();
    //         }
    //     }

    //     void add_linked_edge(std::shared_ptr<EdgeTag> linked_edge)
    //     {
    //         if (m_is_connected)
    //         {
    //             linked_edge->connect();
    //         }

    //         m_linked_edges.emplace_back(std::move(linked_edge));
    //     }

    //     // // Allows keeping a downstream edge holder alive for the lifetime of this edge
    //     // void add_lifetime(std::shared_ptr<EdgeHolder<T>> downstream)
    //     // {
    //     //     m_keep_alive.push_back(downstream);
    //     // }

    //   private:
    //     bool m_is_connected{false};
    //     std::vector<EdgeLifetime> m_connectors;
    //     std::vector<EdgeLifetime> m_disconnectors;
    //     std::vector<std::shared_ptr<EdgeTag>> m_linked_edges;

    // Friend the holder classes which are required to setup connections
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

struct EdgeTypePair
{
  public:
    EdgeTypePair(const EdgeTypePair& other) = default;

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

    bool operator==(const EdgeTypePair& other) const
    {
        return m_is_deferred == other.m_is_deferred && m_full_type == other.m_full_type &&
               m_unwrapped_type == other.m_unwrapped_type;
    }

    template <typename T>
    static EdgeTypePair create()
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

    static EdgeTypePair create_deferred()
    {
        return {std::nullopt, std::nullopt, true};
    }

  private:
    EdgeTypePair(std::optional<std::type_index> full_type,
                 std::optional<std::type_index> unwrapped_type,
                 bool is_deferred) :
      m_full_type(full_type),
      m_unwrapped_type(unwrapped_type),
      m_is_deferred(is_deferred)
    {
        CHECK((m_is_deferred && !m_full_type.has_value() && !m_unwrapped_type.has_value()) ||
              (!m_is_deferred && m_full_type.has_value() && m_unwrapped_type.has_value()))
            << "Inconsistent deferred setting with concrete types";
    }

    std::optional<std::type_index> m_full_type;       // Includes any wrappers like shared_ptr
    std::optional<std::type_index> m_unwrapped_type;  // Excludes any wrappers like shared_ptr if they exist
    bool m_is_deferred{false};                        // Whether or not this type is deferred or concrete
};

struct EdgeHandleObj
{
  public:
    const EdgeTypePair& get_type() const
    {
        return m_type;
    }

  protected:
    EdgeHandleObj(EdgeTypePair type_pair, std::shared_ptr<EdgeTag> edge_handle) :
      m_type(type_pair),
      m_handle(std::move(edge_handle))
    {}

    std::shared_ptr<EdgeTag> get_handle() const
    {
        return m_handle;
    }

    template <typename T>
    std::shared_ptr<T> get_handle_typed() const
    {
        return std::dynamic_pointer_cast<T>(m_handle);
    }

  private:
    EdgeTypePair m_type;

    std::shared_ptr<EdgeTag> m_handle{};

    // // Allow EdgeBuilder to access the internal edge
    // friend EdgeBuilder;

    // Allow ingress and egress derived objects to specialize
    friend IngressHandleObj;
    friend EgressHandleObj;

    // Add EdgeHandle to unpack the object before discarding
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

class IEdgeWritableBase : public virtual EdgeTag
{
  public:
    ~IEdgeWritableBase() override = default;

    virtual EdgeTypePair get_type() const = 0;
};

class IEdgeReadableBase : public virtual EdgeTag
{
  public:
    ~IEdgeReadableBase() override = default;

    virtual EdgeTypePair get_type() const = 0;
};

template <typename T>
class IEdgeWritable : public virtual EdgeHandle<T>, public virtual IEdgeWritableBase
{
  public:
    EdgeTypePair get_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    virtual channel::Status await_write(T&& data) = 0;

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline channel::Status await_write(T data)
    {
        return await_write(std::move(data));
    }
};

// template <typename SourceT, typename SinkT>
// class EdgeConverter : public EdgeWritable<SinkT>
// {
//     inline channel::Status await_write(SourceT&& data) final
//     {
//         return this->ingress().await_write(std::move(data));
//     }
// };

template <typename T>
class IEdgeReadable : public virtual EdgeHandle<T>, public IEdgeReadableBase
{
  public:
    EdgeTypePair get_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    virtual channel::Status await_read(T& t) = 0;
};

template <typename SourceT, typename SinkT = SourceT>
class ConvertingEdgeWritableBase : public IEdgeWritable<SourceT>
{
  public:
    using source_t = SourceT;
    using sink_t   = SinkT;

    ConvertingEdgeWritableBase(std::shared_ptr<IEdgeWritable<SinkT>> downstream) : m_downstream(downstream)
    {
        this->add_linked_edge(downstream);
    }

  protected:
    inline IEdgeWritable<SinkT>& downstream() const
    {
        return *m_downstream;
    }

  private:
    std::shared_ptr<IEdgeWritable<SinkT>> m_downstream{};
};

template <typename SourceT, typename SinkT = SourceT, typename EnableT = void>
class ConvertingEdgeWritable;

template <typename SourceT, typename SinkT>
class ConvertingEdgeWritable<SourceT, SinkT, std::enable_if_t<std::is_convertible_v<SourceT, SinkT>>>
  : public ConvertingEdgeWritableBase<SourceT, SinkT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<SourceT, SinkT>;
    using typename base_t::sink_t;
    using typename base_t::source_t;

    using base_t::ConvertingEdgeWritableBase;

    channel::Status await_write(source_t&& data) override
    {
        return this->downstream().await_write(std::move(data));
    }
};

template <typename SourceT, typename SinkT = SourceT, typename EnableT = void>
class ConvertingEdgeReadable;

template <typename SourceT, typename SinkT>
class ConvertingEdgeReadable<SourceT, SinkT, std::enable_if_t<std::is_convertible_v<SourceT, SinkT>>>
  : public IEdgeReadable<SinkT>
{
  public:
    ConvertingEdgeReadable(std::shared_ptr<IEdgeReadable<SourceT>> upstream) : m_upstream(upstream)
    {
        this->add_linked_edge(upstream);
    }

    channel::Status await_read(SinkT& data) override
    {
        SourceT source_data;
        auto ret_val = this->upstream().await_read(source_data);

        // Convert to the sink type
        data = source_data;

        return ret_val;
    }

  protected:
    inline IEdgeReadable<SourceT>& upstream() const
    {
        return *m_upstream;
    }

  private:
    std::shared_ptr<IEdgeReadable<SourceT>> m_upstream{};
};

template <typename SourceT, typename SinkT>
class LambdaConvertingEdgeWritable : public ConvertingEdgeWritableBase<SourceT, SinkT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<SourceT, SinkT>;
    using typename base_t::sink_t;
    using typename base_t::source_t;
    using lambda_fn_t = std::function<sink_t(source_t&&)>;

    LambdaConvertingEdgeWritable(lambda_fn_t lambda_fn, std::shared_ptr<IEdgeWritable<sink_t>> downstream) :
      ConvertingEdgeWritableBase<source_t, sink_t>(downstream),
      m_lambda_fn(std::move(lambda_fn))
    {}

    channel::Status await_write(source_t&& data) override
    {
        return this->downstream().await_write(m_lambda_fn(std::move(data)));
    }

  private:
    lambda_fn_t m_lambda_fn{};
};

// // EdgeChannel holds an actual channel object and provides interfaces for reading/writing
// template <typename T>
// class EdgeChannel : public EdgeReadable<T>, public EdgeWritable<T>
// {
//   public:
//     EdgeChannel(std::unique_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
//     virtual ~EdgeChannel()
//     {
//         if (m_channel)
//         {
//             m_channel->close_channel();
//             m_channel.reset();
//         }
//     }

//     virtual channel::Status await_read(T& t)
//     {
//         return m_channel->await_read(t);
//     }

//     virtual channel::Status await_write(T&& t)
//     {
//         return m_channel->await_write(std::move(t));
//     }

//   private:
//     std::unique_ptr<mrc::channel::Channel<T>> m_channel;
// };

// template <typename T>
// class EdgeChannelReader : public EdgeReadable<T>
// {
//   public:
//     EdgeChannelReader(std::shared_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
//     virtual ~EdgeChannelReader()
//     {
//         if (m_channel)
//         {
//             if (this->is_connected())
//             {
//                 VLOG(10) << "Closing channel from EdgeChannelReader";
//             }

//             m_channel->close_channel();
//         }
//     }

//     virtual channel::Status await_read(T& t)
//     {
//         return m_channel->await_read(t);
//     }

//   private:
//     std::shared_ptr<mrc::channel::Channel<T>> m_channel;
// };

// template <typename T>
// class EdgeChannelWriter : public EdgeWritable<T>
// {
//   public:
//     EdgeChannelWriter(std::shared_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
//     virtual ~EdgeChannelWriter()
//     {
//         if (m_channel)
//         {
//             if (this->is_connected())
//             {
//                 VLOG(10) << "Closing channel from EdgeChannelWriter";
//             }
//             m_channel->close_channel();
//         }
//     }

//     virtual channel::Status await_write(T&& t)
//     {
//         return m_channel->await_write(std::move(t));
//     }

//   private:
//     std::shared_ptr<mrc::channel::Channel<T>> m_channel;
// };

// EdgeHolder keeps shared pointer of EdgeChannel alive and
template <typename T>
class EdgeHolder : public virtual_enable_shared_from_this<EdgeHolder<T>>
{
  public:
    // void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    // {
    //     // Create a new edge that will close the channel when all
    //     auto new_edge = std::make_shared<EdgeChannel<T>>(std::move(channel));

    //     this->set_edge(new_edge);
    // }
    EdgeHolder()          = default;
    virtual ~EdgeHolder() = default;

  protected:
    void init_edge(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check if set_edge followed by get_edge has been called
        if ((m_owned_edge.lock() && !m_owned_edge_lifetime) || m_edge_connection)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // // Check for existing edge
        // if (m_set_edge)
        // {
        //     if (m_set_edge->is_connected())
        //     {
        //         throw std::runtime_error(
        //             "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
        //     }
        // }

        std::weak_ptr<EdgeHandle<T>> weak_edge = edge;

        edge->add_connector(EdgeLifetime([this, weak_edge]() {
            // Convert to full shared_ptr to avoid edge going out of scope
            if (auto e = weak_edge.lock())
            {
                // this->m_owned_edge_lifetime.reset();

                auto self = this->shared_from_this();

                // Release the lifetime on self
                self->m_owned_edge_lifetime.reset();

                // Now register a disconnector to keep self alive
                e->add_disconnector(EdgeLifetime([self]() {
                    self->m_owned_edge_lifetime.reset();
                    self->m_owned_edge.reset();
                }));
            }
            else
            {
                LOG(ERROR) << "Lock was destroyed before making connection.";
            }
        }));

        // Set to the temp edge to ensure its alive until get_edge is called
        m_owned_edge_lifetime = edge;

        // Set to the weak ptr as well
        m_owned_edge = edge;
    }

    std::shared_ptr<EdgeHandleObj> get_edge_connection() const
    {
        if (auto edge = m_owned_edge.lock())
        {
            return std::shared_ptr<EdgeHandleObj>(new EdgeHandleObj(EdgeTypePair::create<T>(), edge));
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    // std::shared_ptr<EdgeHandle<T>> get_edge_OLD() const
    // {
    //     if (auto edge = m_owned_edge.lock())
    //     {
    //         // // Clear the temp holder edge
    //         // const_cast<EdgeHolder<T>*>(this)->reset_edge();

    //         // auto self = const_cast<EdgeHolder<T>*>(this)->shared_from_this();

    //         // // Add this object to the lifetime of the edge to ensure we are alive while the channel is held
    //         // edge->add_lifetime(self);

    //         return edge;
    //     }

    //     throw std::runtime_error("Must set an edge before calling get_edge");
    // }

    void make_edge_connection(std::shared_ptr<EdgeHandleObj> edge_obj)
    {
        CHECK(edge_obj->get_type() == EdgeTypePair::create<T>())
            << "Incoming edge connection is not the correct type. Make sure to call "
               "`EdgeBuilder::adapt_ingress<T>(edge)` or `EdgeBuilder::adapt_egress<T>(edge)` before calling "
               "make_edge_connection";

        // Unpack the edge, convert, and call the inner set_edge
        auto unpacked_edge = edge_obj->get_handle_typed<EdgeHandle<T>>();

        this->set_edge_handle(unpacked_edge);
    }

    void release_edge_connection()
    {
        m_owned_edge_lifetime.reset();
        m_edge_connection.reset();
    }

    // Used for retrieving the current edge without altering its lifetime
    std::weak_ptr<EdgeHandle<T>> m_owned_edge;

    // Holds a pointer to any set edge (different from init edge). Maintains lifetime
    std::shared_ptr<EdgeHandle<T>> m_edge_connection;

    // This object ensures that any initialized edge is kept alive and is cleared on connection
    std::shared_ptr<EdgeHandle<T>> m_owned_edge_lifetime;

  private:
    void set_edge_handle(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check if set_edge followed by get_edge has been called
        if (m_owned_edge.lock() && !m_owned_edge_lifetime)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // Check for existing edge
        if (m_edge_connection)
        {
            if (m_edge_connection->is_connected())
            {
                throw std::runtime_error(
                    "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
            }
        }

        // Set to the temp edge to ensure its alive until get_edge is called
        m_edge_connection = edge;

        // Set to the weak ptr as well
        m_owned_edge = edge;

        // Remove any init lifetime
        m_owned_edge_lifetime.reset();

        // Now indicate that we have a connection
        edge->connect();
    }

    // Allow edge builder to call set_edge
    friend EdgeBuilder;
};

template <typename T, typename KeyT>
class MultiEdgeHolder : public virtual_enable_shared_from_this<MultiEdgeHolder<T, KeyT>>
{
  public:
    MultiEdgeHolder()          = default;
    virtual ~MultiEdgeHolder() = default;

  protected:
    using edge_pair_t = std::pair<std::weak_ptr<EdgeHandle<T>>, std::shared_ptr<EdgeHandle<T>>>;

    void init_edge(KeyT key, std::shared_ptr<EdgeHandle<T>> edge)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        // Check if set_edge followed by get_edge has been called
        if (edge_pair.first.lock() && !edge_pair.second)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // Check for existing edge
        if (edge_pair.second)
        {
            if (edge_pair.second->is_connected())
            {
                throw std::runtime_error(
                    "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
            }
        }

        std::weak_ptr<EdgeHandle<T>> weak_edge = edge;

        edge->add_connector(EdgeLifetime([this, weak_edge, key]() {
            // Convert to full shared_ptr to avoid edge going out of scope
            if (auto e = weak_edge.lock())
            {
                auto self = this->shared_from_this();

                // Reset the edge on self
                self->reset_edge(key);

                // Now register a disconnector to keep self alive
                e->add_disconnector(EdgeLifetime([self, key]() {
                    auto& ep = self->get_edge_pair(key);

                    ep.first.reset();
                    ep.second.reset();
                }));
            }
            else
            {
                LOG(ERROR) << "Lock was destroyed before making connection.";
            }
        }));

        // Set to the temp edge to ensure its alive until get_edge is called
        edge_pair.second = edge;

        // Set to the weak ptr as well
        edge_pair.first = edge;
    }

    std::shared_ptr<EdgeHandleObj> get_edge_connection(const KeyT& key) const
    {
        auto& edge_pair = this->get_edge_pair(key);

        if (auto edge = edge_pair.first.lock())
        {
            return std::shared_ptr<EdgeHandleObj>(new EdgeHandleObj(EdgeTypePair::create<T>(), edge));
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    void make_edge_connection(KeyT key, std::shared_ptr<EdgeHandleObj> edge_obj)
    {
        CHECK(edge_obj->get_type() == EdgeTypePair::create<T>())
            << "Incoming edge connection is not the correct type. Make sure to call "
               "`EdgeBuilder::adapt_ingress<T>(edge)` or `EdgeBuilder::adapt_egress<T>(edge)` before calling "
               "make_edge_connection";

        // Unpack the edge, convert, and call the inner set_edge
        auto unpacked_edge = edge_obj->get_handle_typed<EdgeHandle<T>>();

        this->set_edge_handle(key, unpacked_edge);
    }

    void release_edge_connection(const KeyT& key)
    {
        auto& edge_pair = this->get_edge_pair(key, true);
        edge_pair.first.reset();
        edge_pair.second.reset();
    }

    void release_edge_connections()
    {
        m_edges.clear();
    }

    size_t edge_connection_count() const
    {
        return m_edges.size();
    }

    std::vector<KeyT> edge_connection_keys() const
    {
        std::vector<KeyT> keys;

        for (const auto& [key, _] : m_edges)
        {
            keys.push_back(key);
        }

        return keys;
    }

    edge_pair_t& get_edge_pair(KeyT key, bool create_if_missing = false)
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            if (create_if_missing)
            {
                m_edges[key] = edge_pair_t();
                return m_edges[key];
            }

            throw std::runtime_error(MRC_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

    const edge_pair_t& get_edge_pair(KeyT key) const
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            throw std::runtime_error(MRC_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

    // Keeps pairs of get_edge/set_edge for each key
    std::map<KeyT, edge_pair_t> m_edges;

  private:
    void set_edge_handle(KeyT key, std::shared_ptr<EdgeHandle<T>> edge)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        // Check if set_edge followed by get_edge has been called
        if (edge_pair.first.lock() && !edge_pair.second)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // Check for existing edge
        if (edge_pair.second)
        {
            if (edge_pair.second->is_connected())
            {
                throw std::runtime_error(
                    "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
            }
        }

        // Set to the temp edge to ensure its alive until get_edge is called
        edge_pair.first = edge;

        // Set to the weak ptr as well
        edge_pair.second = edge;

        // Now indicate that we have a connection
        edge->connect();
    }

    // Allow edge builder to call set_edge
    friend EdgeBuilder;
};

// // Equivalent to SinkProperties
// template <typename T>
// class UpstreamEdgeHolder : public EdgeHolder<T>
// {
//   protected:
//     std::shared_ptr<EdgeReadable<T>> get_readable_edge() const
//     {
//         return std::dynamic_pointer_cast<EdgeReadable<T>>(this->m_set_edge);
//     }
// };

// template <typename T>
// class DownstreamEdgeHolder : public EdgeHolder<T>
// {
//   protected:
//     std::shared_ptr<EdgeWritable<T>> get_writable_edge() const
//     {
//         return std::dynamic_pointer_cast<EdgeWritable<T>>(this->m_set_edge);
//     }
// };

// template <typename T, typename KeyT>
// class DownstreamMultiEdgeHolder : public MultiEdgeHolder<T, KeyT>
// {
//   protected:
//     std::shared_ptr<EdgeWritable<T>> get_writable_edge(KeyT edge_key) const
//     {
//         return std::dynamic_pointer_cast<EdgeWritable<T>>(this->get_edge_pair(edge_key).second);
//     }
// };

// template <typename T>
// class UpstreamChannelHolder : public virtual UpstreamEdgeHolder<T>
// {
//   public:
//     void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
//     {
//         this->do_set_channel(std::move(channel));
//     }

//   protected:
//     void do_set_channel(std::shared_ptr<mrc::channel::Channel<T>> shared_channel)
//     {
//         // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still
//         use
//         // get_readable+edge
//         auto channel_reader = std::make_shared<EdgeChannelReader<T>>(shared_channel);
//         auto channel_writer = std::make_shared<EdgeChannelWriter<T>>(shared_channel);

//         channel_writer->add_connector(EdgeLifetime([this, channel_reader]() {
//             // On connection, save the reader so we can use the channel without it being deleted
//             this->m_set_edge = channel_reader;
//         }));

//         UpstreamEdgeHolder<T>::init_edge(channel_writer);
//     }
// };

// template <typename T>
// class DownstreamChannelHolder : public virtual DownstreamEdgeHolder<T>
// {
//   public:
//     void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
//     {
//         this->do_set_channel(std::move(channel));
//     }

//   protected:
//     void do_set_channel(std::shared_ptr<mrc::channel::Channel<T>> shared_channel)
//     {
//         // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still
//         use
//         // get_writable_edge
//         auto channel_reader = std::make_shared<EdgeChannelReader<T>>(shared_channel);
//         auto channel_writer = std::make_shared<EdgeChannelWriter<T>>(shared_channel);

//         channel_reader->add_connector(EdgeLifetime([this, channel_writer]() {
//             // On connection, save the writer so we can use the channel without it being deleted
//             this->m_set_edge = channel_writer;
//         }));

//         DownstreamEdgeHolder<T>::init_edge(channel_reader);
//     }
// };

struct DeferredIngressHandleObj;

struct IngressHandleObj : public EdgeHandleObj
{
    IngressHandleObj(std::shared_ptr<IEdgeWritableBase> ingress) : IngressHandleObj(ingress->get_type(), ingress) {}

    static std::shared_ptr<IngressHandleObj> from_typeless(std::shared_ptr<EdgeHandleObj> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeWritableBase>();

        CHECK(typed_ingress) << "Could not convert to ingress";

        return std::make_shared<IngressHandleObj>(std::move(typed_ingress));
    }

    virtual bool is_deferred() const
    {
        return false;
    }

  protected:
    // Allow manually specifying the edge type
    IngressHandleObj(EdgeTypePair edge_type, std::shared_ptr<IEdgeWritableBase> ingress) :
      EdgeHandleObj(edge_type, ingress)
    {}

  private:
    std::shared_ptr<IEdgeWritableBase> get_ingress() const
    {
        return std::dynamic_pointer_cast<IEdgeWritableBase>(this->get_handle());
    }

    template <typename T>
    std::shared_ptr<IEdgeWritable<T>> get_ingress_typed() const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_handle());
    }

    void set_ingress_handle(std::shared_ptr<IEdgeWritableBase> ingress)
    {
        this->m_type   = ingress->get_type();
        this->m_handle = ingress;
    }

    // Allow EdgeBuilder to unpack the edge
    friend EdgeBuilder;

    // Add deferred ingresses to set their deferred type
    friend DeferredIngressHandleObj;
};

// class DefaultDeferredEdge : public IEdgeWritableBase
// {
//   public:
//     EdgeTypePair get_type() const override
//     {
//         throw std::runtime_error("Not implemented");
//     }
// };

struct EgressHandleObj : public EdgeHandleObj
{
    EgressHandleObj(std::shared_ptr<IEdgeReadableBase> egress) : EdgeHandleObj(egress->get_type(), egress) {}

    static std::shared_ptr<EgressHandleObj> from_typeless(std::shared_ptr<EdgeHandleObj> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeReadableBase>();

        CHECK(typed_ingress) << "Could not convert to egress";

        return std::make_shared<EgressHandleObj>(std::move(typed_ingress));
    }

  private:
    std::shared_ptr<IEdgeReadableBase> get_egress() const
    {
        return std::dynamic_pointer_cast<IEdgeReadableBase>(this->get_handle());
    }

    friend EdgeBuilder;
};

class IEgressProviderBase
{
  public:
    virtual std::shared_ptr<EdgeTag> get_egress_typeless() const = 0;

    virtual std::shared_ptr<EgressHandleObj> get_egress_obj() const = 0;

    virtual EdgeTypePair egress_provider_type() const = 0;
};

class IEgressAcceptorBase
{
  public:
    virtual void set_egress_typeless(std::shared_ptr<EdgeTag> egress) = 0;

    virtual void set_egress_obj(std::shared_ptr<EgressHandleObj> egress) = 0;

    virtual EdgeTypePair egress_acceptor_type() const = 0;
};

class IIngressProviderBase
{
  public:
    virtual std::shared_ptr<EdgeTag> get_ingress_typeless() const = 0;

    virtual std::shared_ptr<IngressHandleObj> get_ingress_obj() const = 0;

    virtual EdgeTypePair ingress_provider_type() const = 0;
};

class IIngressAcceptorBase
{
  public:
    virtual void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) = 0;

    virtual void set_ingress_obj(std::shared_ptr<IngressHandleObj> ingress) = 0;

    virtual EdgeTypePair ingress_acceptor_type() const = 0;
};

template <typename KeyT>
class IMultiIngressAcceptorBase
{
  public:
    // virtual void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) = 0;

    virtual void set_ingress_obj(KeyT key, std::shared_ptr<IngressHandleObj> ingress) = 0;

    // virtual EdgeTypePair ingress_acceptor_type() const = 0;
};

template <typename T>
class IEgressProvider : public IEgressProviderBase
{
  public:
    // virtual std::shared_ptr<IEdgeReadable<T>> get_egress() const = 0;

    std::shared_ptr<EdgeTag> get_egress_typeless() const override
    {
        return nullptr;
        // return this->get_egress();
    }

    EdgeTypePair egress_provider_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    // std::shared_ptr<EgressHandleObj> get_egress_obj() const override
    // {
    //     return std::make_shared<EgressHandleObj>(this->get_egress());
    // }
};

template <typename T>
class IEgressAcceptor : public IEgressAcceptorBase
{
  public:
    // virtual void set_egress(std::shared_ptr<IEdgeReadable<T>> egress) = 0;

    void set_egress_typeless(std::shared_ptr<EdgeTag> egress) override
    {
        // this->set_egress(std::dynamic_pointer_cast<IEdgeReadable<T>>(egress));
    }

    EdgeTypePair egress_acceptor_type() const override
    {
        return EdgeTypePair::create<T>();
    }
};

template <typename T>
class IIngressProvider : public IIngressProviderBase
{
  public:
    std::shared_ptr<EdgeTag> get_ingress_typeless() const override
    {
        // return std::dynamic_pointer_cast<EdgeTag>(this->get_ingress());
        return nullptr;
    }

    EdgeTypePair ingress_provider_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    // std::shared_ptr<IngressHandleObj> get_ingress_obj() const override
    // {
    //     return std::make_shared<IngressHandleObj>(this->get_ingress());
    // }

    //   private:
    //     virtual std::shared_ptr<IEdgeWritable<T>> get_ingress() const = 0;
};

template <typename T>
class IIngressAcceptor : public IIngressAcceptorBase
{
  public:
    void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) override
    {
        // this->set_ingress(std::dynamic_pointer_cast<IEdgeWritable<T>>(ingress));
    }

    EdgeTypePair ingress_acceptor_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    //   private:
    //     virtual void set_ingress(std::shared_ptr<IEdgeWritable<T>> ingress) = 0;
};

template <typename T, typename KeyT>
class IMultiIngressAcceptor : public IMultiIngressAcceptorBase<KeyT>
{};

// template <typename T>
// class EgressProvider : public IEgressProvider<T>, public virtual DownstreamEdgeHolder<T>
// {
//   public:
//     std::shared_ptr<EdgeReadable<T>> get_egress() const
//     {
//         return std::dynamic_pointer_cast<EdgeReadable<T>>(DownstreamEdgeHolder<T>::get_edge());
//     }
//     // virtual std::shared_ptr<EdgeReadable<T>> get_egress() const = 0;
//   private:
//     using DownstreamEdgeHolder<T>::set_edge;
// };

// template <typename T>
// class EgressAcceptor : public IEgressAcceptor<T>, public virtual UpstreamEdgeHolder<T>
// {
//   public:
//     void set_egress(std::shared_ptr<EdgeReadable<T>> egress)
//     {
//         UpstreamEdgeHolder<T>::set_edge(egress);
//     }

//   private:
//     using UpstreamEdgeHolder<T>::set_edge;
// };

// template <typename T>
// class IngressProvider : public IIngressProvider<T>, public virtual UpstreamEdgeHolder<T>
// {
//   public:
//     std::shared_ptr<EdgeWritable<T>> get_ingress() const
//     {
//         return std::dynamic_pointer_cast<EdgeWritable<T>>(UpstreamEdgeHolder<T>::get_edge());
//     }
//     // virtual std::shared_ptr<EdgeWritable<T>> get_ingress() const = 0;
//   private:
//     using UpstreamEdgeHolder<T>::set_edge;
// };

// template <typename T>
// class IngressAcceptor : public IIngressAcceptor<T>, public virtual DownstreamEdgeHolder<T>
// {
//   public:
//     void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress)
//     {
//         DownstreamEdgeHolder<T>::set_edge(ingress);
//     }

//   private:
//     using DownstreamEdgeHolder<T>::set_edge;
// };

// template <typename T>
// class MultiIngressAcceptor : public IIngressAcceptor<T>, public virtual DownstreamMultiEdgeHolder<T, size_t>
// {
//   public:
//     void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress)
//     {
//         auto count = DownstreamMultiEdgeHolder<T, size_t>::edge_count();
//         DownstreamMultiEdgeHolder<T, size_t>::set_edge(count, ingress);
//     }
// };

// template <typename SourceT, typename SinkT = SourceT>
// void make_edge(EgressProvider<SourceT>& source, EgressAcceptor<SinkT>& sink)
// {
//     // Get the egress from the provider
//     auto egress = source.get_egress();

//     // Set the egress to the acceptor
//     sink.set_egress(egress);
// }

// template <typename SourceT, typename SinkT = SourceT>
// void make_edge(IngressAcceptor<SourceT>& source, IngressProvider<SinkT>& sink)
// {
//     // Get ingress from provider
//     auto ingress = sink.get_ingress();

//     // Set to the acceptor
//     source.set_ingress(ingress);
// }

// template <template <typename> class SourceT,
//           template <typename>
//           class SinkT,
//           typename SourceValueT,
//           typename SinkValueT>
// void make_edge2(SourceT<SourceValueT>& source, SinkT<SinkValueT>& sink)
// {
//     using source_full_t = SourceT<SourceValueT>;
//     using sink_full_t   = SinkT<SinkValueT>;

//     if constexpr (is_base_of_template<IngressAcceptor, source_full_t>::value &&
//                   is_base_of_template<IngressProvider, sink_full_t>::value)
//     {
//         // Get ingress from provider
//         auto ingress = sink.get_ingress();

//         // Set to the acceptor
//         source.set_ingress(ingress);
//     }
//     else
//     {
//         static_assert(!sizeof(source_full_t),
//                       "Arguments to make_edge were incorrect. Ensure you are providing either "
//                       "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
//     }
// }

}  // namespace mrc::node
