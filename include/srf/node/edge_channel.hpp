#pragma once

#include "srf/node/channel_holder.hpp"

#include <memory>

namespace srf::node {

template <typename T>
class EdgeChannelReader;

template <typename T>
class EdgeChannelWriter;

template <typename T>
class EdgeChannel;

template <typename T>
class EdgeChannelReader : public IEdgeReadable<T>
{
  public:
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
    EdgeChannelReader(std::shared_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}

    std::shared_ptr<srf::channel::Channel<T>> m_channel;

    template <typename>
    friend class EdgeChannel;
};

template <typename T>
class EdgeChannelWriter : public IEdgeWritable<T>
{
  public:
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
    EdgeChannelWriter(std::shared_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}

    std::shared_ptr<srf::channel::Channel<T>> m_channel;

    template <typename>
    friend class EdgeChannel;
};

// EdgeChannel holds an actual channel object and provides interfaces for reading/writing
template <typename T>
class EdgeChannel
{
  public:
    EdgeChannel(std::unique_ptr<srf::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
    virtual ~EdgeChannel()
    {
        // if (m_channel)
        // {
        //     m_channel->close_channel();
        //     m_channel.reset();
        // }
    }

    // virtual channel::Status await_read(T& t)
    // {
    //     return m_channel->await_read(t);
    // }

    // virtual channel::Status await_write(T&& t)
    // {
    //     return m_channel->await_write(std::move(t));
    // }

    [[nodiscard]] std::shared_ptr<EdgeChannelReader<T>> get_reader() const
    {
        // struct EnableMakeShared : public EdgeChannelReader<T>
        // {
        //     EnableMakeShared() : EdgeChannelReader<T> {}
        // };

        // return std::make_shared<EnableMakeShared>(m_channel);
        return std::shared_ptr<EdgeChannelReader<T>>(new EdgeChannelReader<T>(m_channel));
    }

    [[nodiscard]] std::shared_ptr<EdgeChannelWriter<T>> get_writer() const
    {
        // return std::make_shared<EdgeChannelWriter<T>>(m_channel);
        return std::shared_ptr<EdgeChannelWriter<T>>(new EdgeChannelWriter<T>(m_channel));
    }

  private:
    std::shared_ptr<srf::channel::Channel<T>> m_channel;
};

}  // namespace srf::node
