/*
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

#include "mrc/channel/status.hpp"
#include "mrc/edge/deferred_edge.hpp"
#include "mrc/edge/edge.hpp"
#include "mrc/edge/edge_holder.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/edge/forward.hpp"  // IWYU pragma: keep
#include "mrc/type_traits.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc::edge {

// IWYU pragma: no_forward_declare mrc::edge::ConvertingEdgeReadable
// IWYU pragma: no_forward_declare mrc::edge::ConvertingEdgeWritable

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
        IWritableAcceptorBase& source,
        IWritableProviderBase& sink,
        std::shared_ptr<IEdgeWritableBase> ingress_handle);

    template <typename T>
    static std::shared_ptr<WritableEdgeHandle> adapt_writable_edge(std::shared_ptr<WritableEdgeHandle> ingress);

    template <typename T>
    static std::shared_ptr<ReadableEdgeHandle> adapt_readable_edge(std::shared_ptr<ReadableEdgeHandle> egress);

    /**
     * @brief Attempt to look-up a registered ingress adapter for the given source type and sink properties. If one
     * exists, use it, otherwise fall back to default.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return
     */
    static std::shared_ptr<IEdgeWritableBase> ingress_for_source_type(std::type_index source_type,
                                                                      IWritableProviderBase& sink,
                                                                      std::shared_ptr<IEdgeWritableBase> ingress_handle);

    static void make_edge_writable_typeless(IWritableAcceptorBase& source,
                                            IWritableProviderBase& sink,
                                            bool allow_narrowing = true);

    static void make_edge_readable_typeless(IReadableProviderBase& source,
                                            IReadableAcceptorBase& sink,
                                            bool allow_narrowing = true);

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_writable(IWritableAcceptor<SourceT>& source, IWritableProvider<SinkT>& sink)
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

        std::shared_ptr<WritableEdgeHandle> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = sink.get_writable_edge_handle();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto sink_typed = sink.get_writable_edge_handle()->template get_ingress_typed<SinkT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeWritable<SourceT, SinkT>>(sink_typed);

            // Repack the object back into the handle
            edge = std::make_shared<WritableEdgeHandle>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        source.set_writable_edge_handle(edge);
    }

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_readable(IReadableProvider<SourceT>& source, IReadableAcceptor<SinkT>& sink)
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

        std::shared_ptr<ReadableEdgeHandle> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = source.get_readable_edge_handle();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto source_typed = source.get_readable_edge_handle()->template get_egress_typed<SourceT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeReadable<SourceT, SinkT>>(source_typed);

            // Repack the object back into the handle
            edge = std::make_shared<ReadableEdgeHandle>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        sink.set_readable_edge_handle(edge);
    }

    template <typename EdgeDataTypeT, typename SourceT, typename SinkT, typename SpliceInputT, typename SpliceOutputT>
    static void splice_edge(SourceT& source, SinkT& sink, SpliceInputT& splice_input, SpliceOutputT& splice_output)
    {
        using source_full_t = SourceT;
        using sink_full_t   = SinkT;

        // Have to jump through some hoops here, mimics what 'writable_acceptor_typed' does, so we get everything
        // aligned correctly.
        if constexpr (is_base_of_template<edge::IWritableAcceptor, source_full_t>::value &&
                      is_base_of_template<edge::IWritableProvider, sink_full_t>::value)
        {
            /*
             * In this case, the source object has accepted a writable edge from the sink.
             *
             * Given: [source [edge_handle]] -> [sink]
             *
             * We will:
             * - Get a reference to the edge_handle the source is holding
             * - Reset the Source edge connection
             * - Create a new edge from the source to our WritableProvider splice node
             * - Set the edge_handle for the WritableAcceptor splice node to the edge_handle from the source
             *
             * This will result in the following:
             *
             * [source[new_edge_handle]] -> [splice_node[old_edge_handle]] -> [sink]
             *
             */
            // We don't need to know the data type of the sink, the source node will have the same data type as the
            // splice node, and we already know the sink can provide an edge for the source's data type.
            // [source] -> [sink] => [[source] -> [splice_node]] -> [sink]
            auto* splice_writable_provider = dynamic_cast<edge::IWritableProvider<EdgeDataTypeT>*>(&splice_input);
            CHECK(splice_writable_provider != nullptr) << "Splice input is not a writable provider";

            auto* splice_writable_acceptor = dynamic_cast<edge::IWritableAcceptor<EdgeDataTypeT>*>(&splice_output);
            CHECK(splice_writable_acceptor != nullptr) << "Splice output is not a writable acceptor";

            auto* writable_acceptor = dynamic_cast<edge::IWritableAcceptor<EdgeDataTypeT>*>(&source);
            CHECK(writable_acceptor != nullptr) << "Source is not a writable acceptor";

            auto* edge_holder_ptr = dynamic_cast<edge::EdgeHolder<EdgeDataTypeT>*>(writable_acceptor);
            if (edge_holder_ptr == nullptr)
            {
                LOG(FATAL) << "Writable acceptor failed to cast to EdgeHolder";
            }

            auto& edge_holder = *edge_holder_ptr;
            CHECK(edge_holder.check_active_connection(false)) << "No active connection to splice into";

            auto edge_handle = edge_holder.get_connected_edge();
            edge_holder.release_edge_connection();

            make_edge_writable(*writable_acceptor, *splice_writable_provider);
            make_edge_writable(*splice_writable_acceptor, sink);
        }
        else if constexpr (is_base_of_template<edge::IReadableProvider, source_full_t>::value &&
                           is_base_of_template<edge::IReadableAcceptor, sink_full_t>::value)
        {
            // We don't need to know the data type of the source, the sink node will have the same data type as the
            // splice node, and we already know the source can provide an edge for the sink's data type.
            // [source] -> [sink] => [source] -> [[splice_node] -> [sink]]
            auto* splice_readable_provider = dynamic_cast<edge::IReadableProvider<EdgeDataTypeT>*>(&splice_input);
            CHECK(splice_readable_provider != nullptr) << "Splice input is not a writable provider";

            auto* splice_readable_acceptor = dynamic_cast<edge::IReadableAcceptor<EdgeDataTypeT>*>(&splice_output);
            CHECK(splice_readable_acceptor != nullptr) << "Splice output is not a writable acceptor";

            auto* readable_acceptor = dynamic_cast<edge::IReadableAcceptor<EdgeDataTypeT>*>(&sink);
            CHECK(readable_acceptor != nullptr) << "Sink is not a writable provider";

            auto* edge_holder_ptr = dynamic_cast<edge::EdgeHolder<EdgeDataTypeT>*>(readable_acceptor);
            if (edge_holder_ptr == nullptr)
            {
                LOG(FATAL) << "Readable acceptor failed to cast to EdgeHolder";
            }

            auto& edge_holder = *edge_holder_ptr;
            CHECK(edge_holder.check_active_connection(false)) << "No active connection to splice into";

            // Grab the Acceptor's edge handle and release it from the Acceptor
            // Make sure we hold the edge handle until the new edge to the splice has been formed.
            // TODO(Devin): Can we double check that the edge handle from the source matches the one from the sink?
            auto edge_handle = edge_holder.get_connected_edge();
            edge_holder.release_edge_connection();

            make_edge_readable(source, *splice_readable_acceptor);
            make_edge_readable(*splice_readable_provider, *readable_acceptor);
        }
        else
        {
            static_assert(!sizeof(source_full_t),
                          "Arguments to splice_edge were incorrect. Ensure you are providing either "
                          "WritableAcceptor->WritableProvider or ReadableProvider->ReadableAcceptor");
        }
    }

  private:
    static std::shared_ptr<WritableEdgeHandle> do_adapt_ingress(const EdgeTypeInfo& target_type,
                                                                std::shared_ptr<WritableEdgeHandle> ingress);

    static std::shared_ptr<ReadableEdgeHandle> do_adapt_egress(const EdgeTypeInfo& target_type,
                                                               std::shared_ptr<ReadableEdgeHandle> egress);
};

template <typename T>
class DeferredWritableMultiEdge : public MultiEdgeHolder<std::size_t, T>,
                                  public IEdgeWritable<T>,
                                  public DeferredWritableMultiEdgeBase
{
  public:
    DeferredWritableMultiEdge(determine_indices_fn_t indices_fn = nullptr,
                              bool deep_copy                    = false,
                              std::string name                  = std::string()) :
      m_indices_fn(std::move(indices_fn)),
      MultiEdgeHolder<std::size_t, T>(std::move(name))
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
    void set_writable_edge_handle(std::size_t key, std::shared_ptr<WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = EdgeBuilder::adapt_writable_edge<T>(ingress);

        MultiEdgeHolder<std::size_t, T>::make_edge_connection(key, adapted_ingress);
    }

    bool m_deep_copy{false};
    determine_indices_fn_t m_indices_fn{};
};

template <typename T>
std::shared_ptr<WritableEdgeHandle> EdgeBuilder::adapt_writable_edge(std::shared_ptr<WritableEdgeHandle> ingress)
{
    // Check if the incoming handle object is dynamic
    if (ingress->is_deferred())
    {
        // Cast to a defferred ingress object
        auto deferred_ingress = std::dynamic_pointer_cast<DeferredWritableEdgeHandle>(ingress);

        CHECK(deferred_ingress) << "Deferred ingress object must derive from DeferredIngressHandleObj";

        auto deferred_edge = std::make_shared<DeferredWritableMultiEdge<T>>();

        // Create a new edge and update the ingress
        // ingress = deferred_ingress->make_deferred_edge<T>();
        ingress = deferred_ingress->set_deferred_edge(deferred_edge);
    }

    auto target_type = EdgeTypeInfo::create<T>();

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
std::shared_ptr<ReadableEdgeHandle> EdgeBuilder::adapt_readable_edge(std::shared_ptr<ReadableEdgeHandle> egress)
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

    auto target_type = EdgeTypeInfo::create<T>();

    // Now try and loop over any egress adaptors for the source
    auto adapted_egress = EdgeBuilder::do_adapt_egress(target_type, egress);

    // Try it again in case we need a source adaptor then a sink adaptor (Short circuits if we are already there)
    adapted_egress = EdgeBuilder::do_adapt_egress(target_type, adapted_egress);

    // Convert if neccessary
    // auto egress_adapted = EdgeBuilder::egress_adapter_for_sink(source, sink, egress);

    // Set to the source
    return adapted_egress;
}

}  // namespace mrc::edge

// Put make edge in the mrc namespace since it is used so often
namespace mrc {

template <typename SourceT, typename SinkT>
void make_edge(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (is_base_of_template<edge::IWritableAcceptor, source_full_t>::value &&
                  is_base_of_template<edge::IWritableProvider, sink_full_t>::value)
    {
        // Call the typed version for ingress provider/acceptor
        edge::EdgeBuilder::make_edge_writable(source, sink);
    }
    else if constexpr (is_base_of_template<edge::IReadableProvider, source_full_t>::value &&
                       is_base_of_template<edge::IReadableAcceptor, sink_full_t>::value)
    {
        // Call the typed version for egress provider/acceptor
        edge::EdgeBuilder::make_edge_readable(source, sink);
    }
    else if constexpr (std::is_base_of_v<edge::IWritableAcceptorBase, source_full_t> &&
                       std::is_base_of_v<edge::IWritableProviderBase, sink_full_t>)
    {
        edge::EdgeBuilder::make_edge_writable_typeless(source, sink);
    }
    else if constexpr (std::is_base_of_v<edge::IReadableProviderBase, source_full_t> &&
                       std::is_base_of_v<edge::IReadableAcceptorBase, sink_full_t>)
    {
        edge::EdgeBuilder::make_edge_readable_typeless(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "WritableAcceptor->WritableProvider or ReadableProvider->ReadableAcceptor");
    }
}

template <typename SourceT, typename SinkT>
void make_edge_typeless(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (std::is_base_of_v<edge::IWritableAcceptorBase, source_full_t> &&
                  std::is_base_of_v<edge::IWritableProviderBase, sink_full_t>)
    {
        edge::EdgeBuilder::make_edge_writable_typeless(source, sink);
    }
    else if constexpr (std::is_base_of_v<edge::IReadableProviderBase, source_full_t> &&
                       std::is_base_of_v<edge::IReadableAcceptorBase, sink_full_t>)
    {
        edge::EdgeBuilder::make_edge_readable_typeless(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "WritableAcceptor->WritableProvider or ReadableProvider->ReadableAcceptor");
    }
}

// template <typename SourceT,
//           typename SinkT,
//           typename = std::enable_if_t<is_base_of_template<IWritableAcceptor, SourceT>::value &&
//                                       is_base_of_template<IWritableProvider, SinkT>::value>>
// SinkT& operator|(SourceT& source, SinkT& sink)
// {
//     make_edge(source, sink);

//     return sink;
// }
}  // namespace mrc
