#pragma once

#include "mrc/engine/segment/ibuilder.hpp"
#include "mrc/segment/forward.hpp"  // IWYU pragma: export
#include "mrc/types.hpp"

#include <functional>
#include <memory>

namespace mrc::internal::segment {
class IBuilder;
}

namespace mrc::segment {

using segment_initializer_fn_t = std::function<void(Builder&)>;
using egress_initializer_t     = std::function<std::shared_ptr<EgressPortBase>(const SegmentAddress&)>;
using ingress_initializer_t    = std::function<std::shared_ptr<IngressPortBase>(const SegmentAddress&)>;
using backend_initializer_fn_t = std::function<void(internal::segment::IBuilder&)>;

}  // namespace mrc::segment
