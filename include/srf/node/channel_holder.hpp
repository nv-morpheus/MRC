#pragma once

#include "srf/channel/channel.hpp"
#include "srf/channel/egress.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/type_traits.hpp"

#include <glog/logging.h>

#include <exception>
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

template <typename T>
class EdgeHandle
{
  public:
    virtual ~EdgeHandle()
    {
        VLOG(10) << "Destroying EdgeHandle";
    };

  protected:
    // Allows keeping a downstream edge holder alive for the lifetime of this edge
    void add_lifetime(std::shared_ptr<EdgeHolder<T>> downstream)
    {
        m_keep_alive.push_back(downstream);
    }

  private:
    std::vector<std::shared_ptr<EdgeHolder<T>>> m_keep_alive;

    friend EdgeHolder<T>;
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
    std::shared_ptr<EdgeHandle<T>> get_edge() const
    {
        if (auto edge = m_get_edge.lock())
        {
            // Clear the temp holder edge
            const_cast<EdgeHolder<T>*>(this)->reset_edge();

            auto self = const_cast<EdgeHolder<T>*>(this)->shared_from_this();

            // Add this object to the lifetime of the edge to ensure we are alive while the channel is held
            edge->add_lifetime(self);

            return edge;
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    void set_edge(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check if set_edge followed by get_edge has been called
        if (m_get_edge.lock() && !m_set_edge)
        {
            // Then someone is using this edge already, cant be changed
            throw std::runtime_error("Cant change edge after a connection has been made");
        }

        // Set to the temp edge to ensure its alive until get_edge is called
        m_set_edge = edge;

        // Set to the weak ptr as well
        m_get_edge = edge;
    }

    void reset_edge()
    {
        m_set_edge.reset();
    }

    // When calling get_edge, this will be converted to a shared_ptr
    std::weak_ptr<EdgeHandle<T>> m_get_edge;
    // When calling set_edge, this will keep the edge alive until get_edge is called
    std::shared_ptr<EdgeHandle<T>> m_set_edge;

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

template <typename T>
class UpstreamChannelHolder : public virtual UpstreamEdgeHolder<T>
{
  public:
    void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
    {
        // Create a new edge that will close the channel when all
        auto new_edge = std::make_shared<EdgeChannel<T>>(std::move(channel));

        UpstreamEdgeHolder<T>::set_edge(new_edge);
    }
};

template <typename T>
class DownstreamChannelHolder : public virtual DownstreamEdgeHolder<T>
{
  public:
    void set_channel(std::unique_ptr<srf::channel::Channel<T>> channel)
    {
        // Create a new edge that will close the channel when all
        auto new_edge = std::make_shared<EdgeChannel<T>>(std::move(channel));

        DownstreamEdgeHolder<T>::set_edge(new_edge);
    }
};

template <typename T>
class EgressProvider : public virtual DownstreamEdgeHolder<T>
{
  public:
    std::shared_ptr<EdgeReadable<T>> get_egress() const
    {
        return std::dynamic_pointer_cast<EdgeReadable<T>>(DownstreamEdgeHolder<T>::get_edge());
    }
    // virtual std::shared_ptr<EdgeReadable<T>> get_egress() const = 0;
};

template <typename T>
class EgressAcceptor : public virtual UpstreamEdgeHolder<T>
{
  public:
    void set_egress(std::shared_ptr<EdgeReadable<T>> egress)
    {
        UpstreamEdgeHolder<T>::set_edge(egress);
    }
};

template <typename T>
class IngressProvider : public virtual UpstreamEdgeHolder<T>
{
  public:
    std::shared_ptr<EdgeWritable<T>> get_ingress() const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(UpstreamEdgeHolder<T>::get_edge());
    }
    // virtual std::shared_ptr<EdgeWritable<T>> get_ingress() const = 0;
};

template <typename T>
class IIngressAcceptor
{
  public:
    virtual void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress) = 0;
};

template <typename T>
class IngressAcceptor : public virtual DownstreamEdgeHolder<T>
{
  public:
    void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress)
    {
        DownstreamEdgeHolder<T>::set_edge(ingress);
    }
};

template <typename SourceT, typename SinkT = SourceT>
void make_edge(EgressProvider<SourceT>& source, EgressAcceptor<SinkT>& sink)
{
    // Get the egress from the provider
    auto egress = source.get_egress();

    // Set the egress to the acceptor
    sink.set_egress(egress);
}

template <typename SourceT, typename SinkT = SourceT>
void make_edge(IngressAcceptor<SourceT>& source, IngressProvider<SinkT>& sink)
{
    // Get ingress from provider
    auto ingress = sink.get_ingress();

    // Set to the acceptor
    source.set_ingress(ingress);
}

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

    if constexpr (is_base_of_template<IngressAcceptor, source_full_t>::value &&
                  is_base_of_template<IngressProvider, sink_full_t>::value)
    {
        // Get ingress from provider
        auto ingress = sink.get_ingress();

        // Set to the acceptor
        source.set_ingress(ingress);
    }
    else if constexpr (is_base_of_template<EgressProvider, source_full_t>::value &&
                       is_base_of_template<EgressAcceptor, sink_full_t>::value)
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
