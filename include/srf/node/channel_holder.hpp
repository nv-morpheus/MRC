#pragma once

#include "srf/channel/channel.hpp"
#include "srf/channel/egress.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/type_traits.hpp"
#include "srf/utils/string_utils.hpp"

#include <glog/logging.h>
#include <sys/types.h>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace srf::node {

struct virtual_enable_shared_from_this_base : public std::enable_shared_from_this<virtual_enable_shared_from_this_base>
{
    virtual ~virtual_enable_shared_from_this_base() {}
};

// This class allows enable_shared_from_this to work for virtual inheritance
template <typename T>
struct virtual_enable_shared_from_this : public virtual virtual_enable_shared_from_this_base
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

template <typename T>
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

template <typename T>
class EdgeHandle
{
  public:
    virtual ~EdgeHandle()
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

    void add_connector(EdgeLifetime<T>&& connector)
    {
        m_connectors.emplace_back(std::move(connector));
    }

    void add_disconnector(EdgeLifetime<T>&& disconnector)
    {
        m_disconnectors.emplace_back(std::move(disconnector));
    }

  protected:
    void connect()
    {
        m_is_connected = true;

        // Clear the connectors to execute them
        m_connectors.clear();
    }

    // // Allows keeping a downstream edge holder alive for the lifetime of this edge
    // void add_lifetime(std::shared_ptr<EdgeHolder<T>> downstream)
    // {
    //     m_keep_alive.push_back(downstream);
    // }

  private:
    bool m_is_connected{false};
    std::vector<EdgeLifetime<T>> m_connectors;
    std::vector<EdgeLifetime<T>> m_disconnectors;
    // std::vector<std::shared_ptr<EdgeHolder<T>>> m_keep_alive;

    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

template <typename T>
class EdgeWritable : public virtual EdgeHandle<T>
{
  public:
    virtual channel::Status await_write(T&& t) = 0;
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
class EdgeReadable : public virtual EdgeHandle<T>
{
  public:
    virtual channel::Status await_read(T& t) = 0;
};

// EdgeChannel holds an actual channel object and provides interfaces for reading/writing
template <typename T>
class EdgeChannel : public EdgeReadable<T>, public EdgeWritable<T>
{
  public:
    EdgeChannel(std::unique_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
    virtual ~EdgeChannel()
    {
        if (m_channel)
        {
            m_channel->close_channel();
            m_channel.reset();
        }
    }

    virtual channel::Status await_read(T& t)
    {
        return m_channel->await_read(t);
    }

    virtual channel::Status await_write(T&& t)
    {
        return m_channel->await_write(std::move(t));
    }

  private:
    std::unique_ptr<srf::channel::Channel<T>> m_channel;
};

template <typename T>
class EdgeChannelReader : public EdgeReadable<T>
{
  public:
    EdgeChannelReader(std::shared_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
    virtual ~EdgeChannelReader()
    {
        if (m_channel)
        {
            if (this->is_connected())
            {
                VLOG(10) << "Closing channel from EdgeChannelReader";
            }

            m_channel->close_channel();
        }
    }

    virtual channel::Status await_read(T& t)
    {
        return m_channel->await_read(t);
    }

  private:
    std::shared_ptr<srf::channel::Channel<T>> m_channel;
};

template <typename T>
class EdgeChannelWriter : public EdgeWritable<T>
{
  public:
    EdgeChannelWriter(std::shared_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
    virtual ~EdgeChannelWriter()
    {
        if (m_channel)
        {
            if (this->is_connected())
            {
                VLOG(10) << "Closing channel from EdgeChannelWriter";
            }
            m_channel->close_channel();
        }
    }

    virtual channel::Status await_write(T&& t)
    {
        return m_channel->await_write(std::move(t));
    }

  private:
    std::shared_ptr<srf::channel::Channel<T>> m_channel;
};

// EdgeHolder keeps shared pointer of EdgeChannel alive and
template <typename T>
class EdgeHolder : public virtual_enable_shared_from_this<EdgeHolder<T>>
{
  public:
    // void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
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
        if ((m_get_edge.lock() && !m_init_edge_lifetime) || m_set_edge)
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

        edge->add_connector(EdgeLifetime<T>([this, weak_edge]() {
            // Convert to full shared_ptr to avoid edge going out of scope
            if (auto e = weak_edge.lock())
            {
                // this->m_init_edge_lifetime.reset();

                auto self = this->shared_from_this();

                // Release the lifetime on self
                self->m_init_edge_lifetime.reset();

                // Now register a disconnector to keep self alive
                e->add_disconnector(EdgeLifetime<T>([self]() {
                    self->m_init_edge_lifetime.reset();
                    self->m_get_edge.reset();
                }));
            }
            else
            {
                LOG(ERROR) << "Lock was destroyed before making connection.";
            }
        }));

        // Set to the temp edge to ensure its alive until get_edge is called
        m_init_edge_lifetime = edge;

        // Set to the weak ptr as well
        m_get_edge = edge;
    }

    std::shared_ptr<EdgeHandle<T>> get_edge() const
    {
        if (auto edge = m_get_edge.lock())
        {
            // // Clear the temp holder edge
            // const_cast<EdgeHolder<T>*>(this)->reset_edge();

            // auto self = const_cast<EdgeHolder<T>*>(this)->shared_from_this();

            // // Add this object to the lifetime of the edge to ensure we are alive while the channel is held
            // edge->add_lifetime(self);

            return edge;
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    void set_edge(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check if set_edge followed by get_edge has been called
        if (m_get_edge.lock() && !m_init_edge_lifetime)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // Check for existing edge
        if (m_set_edge)
        {
            if (m_set_edge->is_connected())
            {
                throw std::runtime_error(
                    "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
            }
        }

        // Set to the temp edge to ensure its alive until get_edge is called
        m_set_edge = edge;

        // Set to the weak ptr as well
        m_get_edge = edge;

        // Remove any init lifetime
        m_init_edge_lifetime.reset();

        // Now indicate that we have a connection
        edge->connect();
    }

    void release_edge()
    {
        m_init_edge_lifetime.reset();
        m_set_edge.reset();
    }

    // Used for retrieving the current edge without altering its lifetime
    std::weak_ptr<EdgeHandle<T>> m_get_edge;

    // Holds a pointer to any set edge (different from init edge). Maintains lifetime
    std::shared_ptr<EdgeHandle<T>> m_set_edge;

    // This object ensures that any initialized edge is kept alive and is cleared on connection
    std::shared_ptr<EdgeHandle<T>> m_init_edge_lifetime;

  private:
    // Allow edge builder to call set_edge
    friend EdgeBuilder;
};

template <typename KeyT, typename T>
class MultiEdgeHolder : public virtual_enable_shared_from_this<MultiEdgeHolder<KeyT, T>>
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

        edge->add_connector(EdgeLifetime<T>([this, weak_edge, key]() {
            // Convert to full shared_ptr to avoid edge going out of scope
            if (auto e = weak_edge.lock())
            {
                auto self = this->shared_from_this();

                // Reset the edge on self
                self->reset_edge(key);

                // Now register a disconnector to keep self alive
                e->add_disconnector(EdgeLifetime<T>([self, key]() {
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

    std::shared_ptr<EdgeHandle<T>> get_edge(const KeyT& key) const
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        if (auto edge = edge_pair.first.lock())
        {
            // // Clear the temp holder edge
            // const_cast<EdgeHolder<T>*>(this)->reset_edge();

            // auto self = const_cast<EdgeHolder<T>*>(this)->shared_from_this();

            // // Add this object to the lifetime of the edge to ensure we are alive while the channel is held
            // edge->add_lifetime(self);

            return edge;
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    void set_edge(KeyT key, std::shared_ptr<EdgeHandle<T>> edge)
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

    void reset_edge(const KeyT& key)
    {
        auto& edge_pair = this->get_edge_pair(key, true);
        edge_pair.second.reset();
    }

    size_t edge_count() const
    {
        return m_edges.size();
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
            else
            {
                throw std::runtime_error(SRF_CONCAT_STR("Could not find edge pair for key: " << key));
            }
        }

        return found->second;
    }

    const edge_pair_t& get_edge_pair(KeyT key) const
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            throw std::runtime_error(SRF_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

    // Keeps pairs of get_edge/set_edge for each key
    std::map<KeyT, edge_pair_t> m_edges;

  private:
    // Allow edge builder to call set_edge
    friend EdgeBuilder;
};

template <typename T>
class UpstreamEdgeHolder : public EdgeHolder<T>
{
  protected:
    std::shared_ptr<EdgeReadable<T>> get_readable_edge() const
    {
        return std::dynamic_pointer_cast<EdgeReadable<T>>(this->m_set_edge);
    }
};

template <typename T>
class DownstreamEdgeHolder : public EdgeHolder<T>
{
  protected:
    std::shared_ptr<EdgeWritable<T>> get_writable_edge() const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(this->m_set_edge);
    }
};

template <typename T, typename KeyT>
class DownstreamMultiEdgeHolder : public MultiEdgeHolder<KeyT, T>
{
  protected:
    std::shared_ptr<EdgeWritable<T>> get_writable_edge(KeyT edge_key) const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(this->get_edge_pair(edge_key).second);
    }
};

template <typename T>
class UpstreamChannelHolder : public virtual UpstreamEdgeHolder<T>
{
  public:
    void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
    {
        this->do_set_channel(std::move(channel));
    }

  protected:
    void do_set_channel(std::shared_ptr<srf::channel::Channel<T>> shared_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_readable+edge
        auto channel_reader = std::make_shared<EdgeChannelReader<T>>(shared_channel);
        auto channel_writer = std::make_shared<EdgeChannelWriter<T>>(shared_channel);

        channel_writer->add_connector(EdgeLifetime<T>([this, channel_reader]() {
            // On connection, save the reader so we can use the channel without it being deleted
            this->m_set_edge = channel_reader;
        }));

        UpstreamEdgeHolder<T>::init_edge(channel_writer);
    }
};

template <typename T>
class DownstreamChannelHolder : public virtual DownstreamEdgeHolder<T>
{
  public:
    void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
    {
        this->do_set_channel(std::move(channel));
    }

  protected:
    void do_set_channel(std::shared_ptr<srf::channel::Channel<T>> shared_channel)
    {
        // Create 2 edges, one for reading and writing. On connection, persist the other to allow the node to still use
        // get_writable_edge
        auto channel_reader = std::make_shared<EdgeChannelReader<T>>(shared_channel);
        auto channel_writer = std::make_shared<EdgeChannelWriter<T>>(shared_channel);

        channel_reader->add_connector(EdgeLifetime<T>([this, channel_writer]() {
            // On connection, save the writer so we can use the channel without it being deleted
            this->m_set_edge = channel_writer;
        }));

        DownstreamEdgeHolder<T>::init_edge(channel_reader);
    }
};

template <typename T>
class IEgressProvider
{
  public:
    virtual std::shared_ptr<EdgeReadable<T>> get_egress() const = 0;
};

template <typename T>
class IEgressAcceptor
{
  public:
    virtual void set_egress(std::shared_ptr<EdgeReadable<T>> egress) = 0;
};

template <typename T>
class IIngressProvider
{
  public:
    virtual std::shared_ptr<EdgeWritable<T>> get_ingress() const = 0;
};

template <typename T>
class IIngressAcceptor
{
  public:
    virtual void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress) = 0;
};

template <typename T>
class EgressProvider : public IEgressProvider<T>, public virtual DownstreamEdgeHolder<T>
{
  public:
    std::shared_ptr<EdgeReadable<T>> get_egress() const
    {
        return std::dynamic_pointer_cast<EdgeReadable<T>>(DownstreamEdgeHolder<T>::get_edge());
    }
    // virtual std::shared_ptr<EdgeReadable<T>> get_egress() const = 0;
  private:
    using DownstreamEdgeHolder<T>::set_edge;
};

template <typename T>
class EgressAcceptor : public IEgressAcceptor<T>, public virtual UpstreamEdgeHolder<T>
{
  public:
    void set_egress(std::shared_ptr<EdgeReadable<T>> egress)
    {
        UpstreamEdgeHolder<T>::set_edge(egress);
    }

  private:
    using UpstreamEdgeHolder<T>::set_edge;
};

template <typename T>
class IngressProvider : public IIngressProvider<T>, public virtual UpstreamEdgeHolder<T>
{
  public:
    std::shared_ptr<EdgeWritable<T>> get_ingress() const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(UpstreamEdgeHolder<T>::get_edge());
    }
    // virtual std::shared_ptr<EdgeWritable<T>> get_ingress() const = 0;
  private:
    using UpstreamEdgeHolder<T>::set_edge;
};

template <typename T>
class IngressAcceptor : public IIngressAcceptor<T>, public virtual DownstreamEdgeHolder<T>
{
  public:
    void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress)
    {
        DownstreamEdgeHolder<T>::set_edge(ingress);
    }

  private:
    using DownstreamEdgeHolder<T>::set_edge;
};

template <typename T>
class MultiIngressAcceptor : public IIngressAcceptor<T>, public virtual DownstreamMultiEdgeHolder<T, size_t>
{
  public:
    void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress)
    {
        auto count = DownstreamMultiEdgeHolder<T, size_t>::edge_count();
        DownstreamMultiEdgeHolder<T, size_t>::set_edge(count, ingress);
    }
};

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

template <template <typename> class SourceT,
          template <typename>
          class SinkT,
          typename SourceValueT,
          typename SinkValueT>
void make_edge2(SourceT<SourceValueT>& source, SinkT<SinkValueT>& sink)
{
    using source_full_t = SourceT<SourceValueT>;
    using sink_full_t   = SinkT<SinkValueT>;

    if constexpr (is_base_of_template<IngressAcceptor, source_full_t>::value &&
                  is_base_of_template<IngressProvider, sink_full_t>::value)
    {
        // Get ingress from provider
        auto ingress = sink.get_ingress();

        // Set to the acceptor
        source.set_ingress(ingress);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

template <typename SourceT, typename SinkT>
void make_edge2(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (is_base_of_template<IIngressAcceptor, source_full_t>::value &&
                  is_base_of_template<IIngressProvider, sink_full_t>::value)
    {
        // Get ingress from provider
        auto ingress = sink.get_ingress();

        // Set to the acceptor
        source.set_ingress(ingress);
    }
    else if constexpr (is_base_of_template<IEgressProvider, source_full_t>::value &&
                       is_base_of_template<IEgressAcceptor, sink_full_t>::value)
    {
        // Get the egress from the provider
        auto egress = source.get_egress();

        // Set the egress to the acceptor
        sink.set_egress(egress);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

}  // namespace srf::node
