#pragma once

#include "srf/node/source_properties.hpp"

namespace srf::node {

template <typename T>
class ReadableSubject : public EgressAcceptor<T>
{
  public:
    channel::Status await_read(T& data)
    {
        return this->get_readable_edge()->await_read(data);
    }
};

}  // namespace srf::node
