#pragma once

#pragma once

#include <pysrf/types.hpp>  // IWYU pragma: keep
#include <pysrf/utils.hpp>

#include <srf/channel/forward.hpp>
#include <srf/channel/ingress.hpp>
#include <srf/channel/status.hpp>
#include <srf/node/edge.hpp>
#include <srf/node/edge_adaptor.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/edge_connector.hpp>
#include <srf/node/edge_registry.hpp>
#include <srf/node/forward.hpp>
#include <srf/node/rx_node.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/rx_source.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <functional>  // for function, ref
#include <memory>      // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast, allocator, make_shared
#include <ostream>     // for operator<<
#include <typeindex>   // for type_index
#include <utility>     // for move, forward

namespace srf::pysrf {

/*
struct SinkHelper
{
    using ftype = std::function<std::shared_ptr<channel::IngressHandle>(srf::node::SourcePropertiesBase&,
                                                                        srf::node::SinkPropertiesBase&)>;

    template <typename InputT>
    static ftype f_builder()
    {
        return [](srf::node::SourcePropertiesBase& source_base, srf::node::SinkPropertiesBase& sink_base) {
            auto source_type                                       = source_base.source_type();
            auto sink_type                                         = sink_base.sink_type();
            std::shared_ptr<channel::IngressHandle> ingress_handle = sink_base.ingress_handle();

            if (source_type == typeid(PyHolder))
            {
                // Check to see if we have a conversion in pybind11
                if (pybind11::detail::get_type_info(sink_base.sink_type(true), false))
                {
                    // Shortcut the check to the registered converters
                    auto edge = std::make_shared<node::Edge<PyHolder, InputT>>(
                        std::dynamic_pointer_cast<channel::Ingress<InputT>>(sink_base.ingress_handle()));

                    // Using auto here confuses the lambda's return type with what's returned from
                    // ingress_for_source_type
                    std::shared_ptr<channel::IngressHandle> handle =
                        std::dynamic_pointer_cast<channel::Ingress<PyHolder>>(edge);
                    CHECK(handle);
                    return handle;
                }
            }

            auto fn_converter =
                srf::node::EdgeRegistry::find_converter(source_base.source_type(), sink_base.sink_type());
            return fn_converter(ingress_handle);
            // return srf::node::SinkTestHelper::ingress_for_source_type(source_type, sink_type, ingress_handle);
        };
    }
};
 */

#pragma GCC visibility push(default)
struct PysrfEdgeAdaptor : public srf::node::EdgeAdaptorBase
{
    static std::shared_ptr<channel::IngressHandle> try_construct_ingress_fallback(
        std::type_index source_type,
        srf::node::SinkPropertiesBase& sink_base,
        std::shared_ptr<channel::IngressHandle> ingress_handle);

    std::shared_ptr<channel::IngressHandle> try_construct_ingress(
        srf::node::SourcePropertiesBase& source,
        srf::node::SinkPropertiesBase& sink,
        std::shared_ptr<channel::IngressHandle> ingress_handle) override;
};
#pragma GCC visibility pop

}  // namespace srf::pysrf