#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"

namespace srf::node {

template <typename DataT, typename NodeT>
class ForwardingEdge : public EdgeWritable<int>
{
  public:
    ForwardingEdge(NodeT& parent) : m_parent(parent) {}

    ~ForwardingEdge()
    {
        m_parent.on_complete();
    }

    channel::Status await_write(DataT&& t) override
    {
        return m_parent.on_next(std::move(t));
    }

  private:
    NodeT& m_parent;
};

template <typename InputT, typename OutputT>
class NodeComponent : public ForwardingIngressProvider<InputT>, public IngressAcceptor<OutputT>
{
  public:
    NodeComponent() : ForwardingIngressProvider<InputT>() {}

    virtual ~NodeComponent() = default;

  protected:
    // // Derive from the forwarding edge so it can call protected members
    // class NodeComponentEdge : public ForwardingEdge<InputT, NodeComponent<InputT, OutputT>>
    // {};

    // virtual channel::Status on_next(InputT&& t) = 0;

    void on_complete() override
    {
        SourceProperties<OutputT>::release_edge();
    }
};

template <typename T>
class NodeComponent<T, T> : public ForwardingIngressProvider<T>, public IngressAcceptor<T>
{
  public:
    NodeComponent() : ForwardingIngressProvider<T>() {}

    virtual ~NodeComponent() = default;

  protected:
    channel::Status on_next(int&& t)
    {
        return this->get_writable_edge()->await_write(t);
    }

    void on_complete()
    {
        SourceProperties<T>::release_edge();
    }
};
}  // namespace srf::node
