#pragma once

#include "srf/node/source_properties.hpp"
namespace srf::node {

template <typename T>
class WritableSubject : public IngressAcceptor<T>
{
  public:
    channel::Status await_write(T&& data)
    {
        return this->get_writable_edge()->await_write(std::move(data));
    }
};

}  // namespace srf::node
