/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mrc/node/channel_holder.hpp"
#include "mrc/node/deferred_edge.hpp"
#include "mrc/node/forward.hpp"  // IWYU pragma: keep

#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <typeindex>

namespace mrc::channel {
struct IngressHandle;
}  // namespace mrc::channel

namespace mrc::node {
class SinkPropertiesBase;
class SourcePropertiesBase;
template <typename T>
class ChannelAcceptor;
template <typename T>
class ChannelProvider;
template <typename T>
class SinkProperties;
template <typename T>
class SourceProperties;

// IWYU pragma: no_forward_declare mrc::node::Edge

struct EdgeBuilder final
{
    /**
     * @brief Attempt to look-up a registered ingress adapter given the source and sink properties. If one exists
     * use it, otherwise fall back to the default adapter lookup.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return Ingress handle constructed by the adapter
     */
    static std::shared_ptr<IEdgeWritableBase> ingress_adapter_for_sink(
        IIngressAcceptorBase& source,
        IIngressProviderBase& sink,
        std::shared_ptr<IEdgeWritableBase> ingress_handle);

    template <typename T>
    static std::shared_ptr<IngressHandleObj> adapt_ingress(std::shared_ptr<IngressHandleObj> ingress);

    template <typename T>
    static std::shared_ptr<EgressHandleObj> adapt_egress(std::shared_ptr<EgressHandleObj> egress);

    /**
     * @brief Attempt to look-up a registered ingress adapter for the given source type and sink properties. If one
     * exists, use it, otherwise fall back to default.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return
     */
    static std::shared_ptr<IEdgeWritableBase> ingress_for_source_type(std::type_index source_type,
                                                                      IIngressProviderBase& sink,
                                                                      std::shared_ptr<IEdgeWritableBase> ingress_handle);

    static void make_edge_ingress_typeless(IIngressAcceptorBase& source,
                                           IIngressProviderBase& sink,
                                           bool allow_narrowing = true);

    static void make_edge_egress_typeless(IEgressProviderBase& source,
                                          IEgressAcceptorBase& sink,
                                          bool allow_narrowing = true);

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_ingress(IIngressAcceptor<SourceT>& source, IIngressProvider<SinkT>& sink)
    {
        constexpr bool IsConvertable = std::is_convertible_v<SourceT, SinkT>;
        constexpr bool LessBits      = sizeof(SourceT) > sizeof(SinkT);  // Sink requires more bits than source.
        constexpr bool FloatToInt    = std::is_floating_point_v<SourceT> && std::is_integral_v<SinkT>;  // float -> int
        constexpr bool SignedToUnsigned = std::is_signed_v<SourceT> && !std::is_signed_v<SinkT>;  // signed -> unsigned
        constexpr bool UnsignedToSignedLessBits = !std::is_signed_v<SourceT> && std::is_signed_v<SinkT> &&
                                                  (sizeof(SourceT) == sizeof(SinkT));  // Unsigned component could
                                                                                       // exceed signed limits

        // If its convertable but may result in loss of data, it requires narrowing
        constexpr bool RequiresNarrowing = IsConvertable &&
                                           (LessBits || FloatToInt || SignedToUnsigned || UnsignedToSignedLessBits);

        std::shared_ptr<IngressHandleObj> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = sink.get_ingress_obj();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto sink_typed = sink.get_ingress_obj()->template get_ingress_typed<SinkT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeWritable<SourceT, SinkT>>(sink_typed);

            // Repack the object back into the handle
            edge = std::make_shared<IngressHandleObj>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        source.set_ingress_obj(edge);
    }

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_egress(IEgressProvider<SourceT>& source, IEgressAcceptor<SinkT>& sink)
    {
        constexpr bool IsConvertable = std::is_convertible_v<SinkT, SourceT>;
        constexpr bool LessBits      = sizeof(SinkT) > sizeof(SourceT);  // Sink requires more bits than source.
        constexpr bool FloatToInt    = std::is_floating_point_v<SourceT> && std::is_integral_v<SinkT>;  // float -> int
        constexpr bool SignedToUnsigned = std::is_signed_v<SinkT> && !std::is_signed_v<SourceT>;  // signed -> unsigned
        constexpr bool UnsignedToSignedLessBits = !std::is_signed_v<SinkT> && std::is_signed_v<SourceT> &&
                                                  (sizeof(SourceT) == sizeof(SinkT));  // Unsigned component could
                                                                                       // exceed signed limits

        // If its convertable but may result in loss of data, it requires narrowing
        constexpr bool RequiresNarrowing = IsConvertable &&
                                           (LessBits || FloatToInt || SignedToUnsigned || UnsignedToSignedLessBits);

        std::shared_ptr<EgressHandleObj> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = source.get_egress_obj();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto source_typed = source.get_egress_obj()->template get_egress_typed<SourceT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeReadable<SourceT, SinkT>>(source_typed);

            // Repack the object back into the handle
            edge = std::make_shared<EgressHandleObj>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        sink.set_egress_obj(edge);
    }

  private:
    static std::shared_ptr<IngressHandleObj> do_adapt_ingress(const EdgeTypePair& target_type,
                                                              std::shared_ptr<IngressHandleObj> ingress);

    static std::shared_ptr<EgressHandleObj> do_adapt_egress(const EdgeTypePair& target_type,
                                                            std::shared_ptr<EgressHandleObj> egress);
};

template <typename SourceT, typename SinkT>
void make_edge(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (is_base_of_template<IIngressAcceptor, source_full_t>::value &&
                  is_base_of_template<IIngressProvider, sink_full_t>::value)
    {
        // Call the typed version for ingress provider/acceptor
        EdgeBuilder::make_edge_ingress(source, sink);
    }
    else if constexpr (is_base_of_template<IEgressProvider, source_full_t>::value &&
                       is_base_of_template<IEgressAcceptor, sink_full_t>::value)
    {
        // Call the typed version for egress provider/acceptor
        EdgeBuilder::make_edge_egress(source, sink);
    }
    else if constexpr (std::is_base_of_v<IIngressAcceptorBase, source_full_t> &&
                       std::is_base_of_v<IIngressProviderBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_ingress_typeless(source, sink);
    }
    else if constexpr (std::is_base_of_v<IEgressProviderBase, source_full_t> &&
                       std::is_base_of_v<IEgressAcceptorBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_egress_typeless(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

template <typename SourceT, typename SinkT>
void make_edge_typeless(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (std::is_base_of_v<IIngressAcceptorBase, source_full_t> &&
                  std::is_base_of_v<IIngressProviderBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_ingress_typeless(source, sink);
    }
    else if constexpr (std::is_base_of_v<IEgressProviderBase, source_full_t> &&
                       std::is_base_of_v<IEgressAcceptorBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_egress_typeless(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

template <typename T>
class DeferredWritableMultiEdge : public MultiEdgeHolder<std::size_t, T>,
                                  public IEdgeWritable<T>,
                                  public DeferredWritableMultiEdgeBase
{
  public:
    DeferredWritableMultiEdge(determine_indices_fn_t indices_fn = nullptr, bool deep_copy = false) :
      m_indices_fn(std::move(indices_fn))
    {
        // // Generate warning if deep_copy = True but type does not support it
        // if constexpr (!std::is_copy_constructible_v<T>)
        // {
        //     if (m_deep_copy)
        //     {
        //         LOG(WARNING) << "DeferredWritableMultiEdge(deep_copy=True) created for type '" << type_name<T>()
        //                      << "' but the type is not copyable. Deep copy will be disabled";

        //         m_deep_copy = false;
        //     }
        // }

        // Set a connector to check that the indices function has been set
        this->add_connector([this]() {
            // Ensure that the indices function is properly set
            CHECK(this->m_indices_fn) << "Must set indices function before connecting edge";
        });
    }

    channel::Status await_write(T&& data) override
    {
        auto indices = this->determine_indices_for_value(data);

        // First, handle the situation where there is more than one connection to push to
        if constexpr (!std::is_copy_constructible_v<T>)
        {
            CHECK(indices.size() <= 1) << type_name<DeferredWritableMultiEdge<T>>()
                                       << " is trying to write to multiple downstreams but the object type is not "
                                          "copyable. Must use copyable type with multiple downstream connections";
        }
        else
        {
            for (size_t i = indices.size() - 1; i > 0; --i)
            {
                // if constexpr (is_shared_ptr<T>::value)
                // {
                //     if (m_deep_copy)
                //     {
                //         auto deep_copy = std::make_shared<typename T::element_type>(*data);
                //         CHECK(this->get_writable_edge(indices[i])->await_write(std::move(deep_copy)) ==
                //               channel::Status::success);
                //         continue;
                //     }
                // }

                T shallow_copy(data);
                CHECK(this->get_writable_edge(indices[i])->await_write(std::move(shallow_copy)) ==
                      channel::Status::success);
            }
        }

        // Always push the last one the same way
        if (indices.size() >= 1)
        {
            return this->get_writable_edge(indices[0])->await_write(std::move(data));
        }

        return channel::Status::success;
    }

    void set_indices_fn(determine_indices_fn_t indices_fn) override
    {
        m_indices_fn = std::move(indices_fn);
    }

    size_t edge_connection_count() const override
    {
        return MultiEdgeHolder<std::size_t, T>::edge_connection_count();
    }
    std::vector<std::size_t> edge_connection_keys() const override
    {
        return MultiEdgeHolder<std::size_t, T>::edge_connection_keys();
    }

  protected:
    std::shared_ptr<IEdgeWritable<T>> get_writable_edge(std::size_t edge_idx) const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_connected_edge(edge_idx));
    }

    virtual std::vector<std::size_t> determine_indices_for_value(const T& data)
    {
        return m_indices_fn(*this);
    }

  private:
    void set_ingress_obj(std::size_t key, std::shared_ptr<IngressHandleObj> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = EdgeBuilder::adapt_ingress<T>(ingress);

        MultiEdgeHolder<std::size_t, T>::make_edge_connection(key, adapted_ingress);
    }

    bool m_deep_copy{false};
    determine_indices_fn_t m_indices_fn{};
};

template <typename T>
std::shared_ptr<IngressHandleObj> EdgeBuilder::adapt_ingress(std::shared_ptr<IngressHandleObj> ingress)
{
    // Check if the incoming handle object is dynamic
    if (ingress->is_deferred())
    {
        // Cast to a defferred ingress object
        auto deferred_ingress = std::dynamic_pointer_cast<DeferredIngressHandleObj>(ingress);

        CHECK(deferred_ingress) << "Deferred ingress object must derive from DeferredIngressHandleObj";

        auto deferred_edge = std::make_shared<DeferredWritableMultiEdge<T>>();

        // Create a new edge and update the ingress
        // ingress = deferred_ingress->make_deferred_edge<T>();
        ingress = deferred_ingress->set_deferred_edge(deferred_edge);
    }

    auto target_type = EdgeTypePair::create<T>();

    // Now try and loop over any ingress adaptors for the sink
    auto adapted_ingress = EdgeBuilder::do_adapt_ingress(target_type, ingress);

    // Try it again in case we need a sink adaptor then a source adaptor (Short circuits if we are already there)
    adapted_ingress = EdgeBuilder::do_adapt_ingress(target_type, adapted_ingress);

    // Convert if neccessary
    // auto ingress_adapted = EdgeBuilder::ingress_adapter_for_sink(source, sink, ingress);

    // Set to the source
    return adapted_ingress;
}

template <typename T>
std::shared_ptr<EgressHandleObj> EdgeBuilder::adapt_egress(std::shared_ptr<EgressHandleObj> egress)
{
    // // Check if the incoming handle object is dynamic
    // if (egress->is_deferred())
    // {
    //     // Cast to a defferred ingress object
    //     auto deferred_ingress = std::dynamic_pointer_cast<DeferredEgressHandleObj>(egress);

    //     CHECK(deferred_ingress) << "Deferred ingress object must derive from DeferredEgressHandleObj";

    //     auto deferred_edge = std::make_shared<DeferredReadableMultiEdge<T>>();

    //     // Create a new edge and update the ingress
    //     // ingress = deferred_ingress->make_deferred_edge<T>();
    //     egress = deferred_ingress->set_deferred_edge(deferred_edge);
    // }

    auto target_type = EdgeTypePair::create<T>();

    // Now try and loop over any egress adaptors for the source
    auto adapted_egress = EdgeBuilder::do_adapt_egress(target_type, egress);

    // Try it again in case we need a source adaptor then a sink adaptor (Short circuits if we are already there)
    adapted_egress = EdgeBuilder::do_adapt_egress(target_type, adapted_egress);

    // Convert if neccessary
    // auto egress_adapted = EdgeBuilder::egress_adapter_for_sink(source, sink, egress);

    // Set to the source
    return adapted_egress;
}

}  // namespace mrc::node
