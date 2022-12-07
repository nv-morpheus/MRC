#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

namespace mrc::node {

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

}  // namespace mrc::node
