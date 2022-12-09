#pragma once

#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

namespace mrc::node {

template <typename InputT, typename OutputT = InputT>
class NodeComponent;

template <typename InputT, typename OutputT>
class NodeComponent : public ForwardingIngressProvider<InputT>, public IngressAcceptor<OutputT>
{
  public:
    NodeComponent() : ForwardingIngressProvider<InputT>() {}

    virtual ~NodeComponent() = default;

  protected:
    void on_complete() override
    {
        SourceProperties<OutputT>::release_edge_connection();
    }
};

template <typename T>
class NodeComponent<T, T> : public ForwardingIngressProvider<T>, public IngressAcceptor<T>
{
  public:
    NodeComponent() : ForwardingIngressProvider<T>() {}

    virtual ~NodeComponent() = default;

  protected:
    channel::Status on_next(T&& t)
    {
        return this->get_writable_edge()->await_write(std::move(t));
    }

    void on_complete()
    {
        SourceProperties<T>::release_edge_connection();
    }
};

template <typename T>
class LambdaNodeComponent : public NodeComponent<T, T>
{
  public:
    using on_next_fn_t     = std::function<T(T&&)>;
    using on_complete_fn_t = std::function<void()>;

    LambdaNodeComponent(on_next_fn_t on_next_fn) : NodeComponent<T, T>(), m_on_next_fn(std::move(on_next_fn)) {}

    LambdaNodeComponent(on_next_fn_t on_next_fn, on_complete_fn_t on_complete_fn) :
      NodeComponent<T, T>(),
      m_on_next_fn(std::move(on_next_fn)),
      m_on_complete_fn(std::move(on_complete_fn))
    {}

    virtual ~LambdaNodeComponent() = default;

  protected:
    channel::Status on_next(T&& t)
    {
        return this->get_writable_edge()->await_write(m_on_next_fn(std::move(t)));
    }

    void on_complete() override
    {
        if (m_on_complete_fn)
        {
            m_on_complete_fn();
        }

        SourceProperties<T>::release_edge_connection();
    }

  private:
    on_next_fn_t m_on_next_fn;
    on_complete_fn_t m_on_complete_fn;
};

}  // namespace mrc::node
