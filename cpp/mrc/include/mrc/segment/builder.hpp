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

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/engine/segment/ibuilder.hpp"  // IWYU pragma: export
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"    // IWYU pragma: export
#include "mrc/node/source_properties.hpp"  // IWYU pragma: export
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/runnable.hpp"  // IWYU pragma: export
#include "mrc/segment/component.hpp"  // IWYU pragma: export
#include "mrc/segment/concepts/object_traits.hpp"
#include "mrc/segment/object.hpp"    // IWYU pragma: export
#include "mrc/segment/runnable.hpp"  // IWYU pragma: export
#include "mrc/type_traits.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/hana/core/when.hpp>  // IWYU pragma: export
#include <boost/hana/if.hpp>         // IWYU pragma: export
#include <boost/hana/type.hpp>       // IWYU pragma: export
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc::node {
template <typename T>
class RxSinkBase;

template <typename T>
class RxSourceBase;
}  // namespace mrc::node

namespace mrc {
struct WatcherInterface;
}  // namespace mrc
namespace mrc::modules {
class PersistentModule;

class SegmentModule;
}  // namespace mrc::modules
namespace mrc::segment {
class Definition;
}  // namespace mrc::segment

namespace {
namespace hana = boost::hana;

template <typename T>
auto has_source_add_watcher = hana::is_valid(
    [](auto&& thing) -> decltype(std::forward<decltype(thing)>(thing).source_add_watcher(
                         std::declval<std::shared_ptr<mrc::WatcherInterface>>())) {});

template <typename T>
auto has_sink_add_watcher = hana::is_valid(
    [](auto&& thing) -> decltype(std::forward<decltype(thing)>(thing).sink_add_watcher(
                         std::declval<std::shared_ptr<mrc::WatcherInterface>>())) {});

template <typename T>
void add_stats_watcher_if_rx_source(T& thing, std::string name)
{
    return hana::if_(
        has_source_add_watcher<T>(thing),
        [name](auto&& object) {
            auto trace_stats = mrc::benchmarking::TraceStatistics::get_or_create(name);
            std::forward<decltype(object)>(object).source_add_watcher(trace_stats);
        },
        [name]([[maybe_unused]] auto&& object) {})(thing);
}

template <typename T>
void add_stats_watcher_if_rx_sink(T& thing, std::string name)
{
    return hana::if_(
        has_sink_add_watcher<T>(thing),
        [name](auto&& object) {
            auto trace_stats = mrc::benchmarking::TraceStatistics::get_or_create(name);
            std::forward<decltype(object)>(object).sink_add_watcher(trace_stats);
        },
        [name]([[maybe_unused]] auto&& object) {})(thing);
}
}  // namespace

namespace mrc::segment {

class Builder final
{
    Builder(internal::segment::IBuilder& backend) : m_backend(backend) {}

  public:
    DELETE_COPYABILITY(Builder);
    DELETE_MOVEABILITY(Builder);

    std::shared_ptr<ObjectProperties> get_ingress(std::string name, std::type_index type_index);

    std::shared_ptr<ObjectProperties> get_egress(std::string name, std::type_index type_index);

    template <typename T>
    std::shared_ptr<Object<node::RxSinkBase<T>>> get_egress(std::string name);

    template <typename T>
    std::shared_ptr<Object<node::RxSourceBase<T>>> get_ingress(std::string name);

    template <typename ObjectT>
    std::shared_ptr<Object<ObjectT>> make_object(std::string name, std::unique_ptr<ObjectT> node);

    template <typename ObjectT, typename... ArgsT>
    std::shared_ptr<Object<ObjectT>> construct_object(std::string name, ArgsT&&... args)
    {
        auto ns_name = m_namespace_prefix.empty() ? name : m_namespace_prefix + "/" + name;
        auto uptr    = std::make_unique<ObjectT>(std::forward<ArgsT>(args)...);

        ::add_stats_watcher_if_rx_source(*uptr, ns_name);
        ::add_stats_watcher_if_rx_sink(*uptr, ns_name);

        return make_object(std::move(ns_name), std::move(uptr));
    }

    template <typename SourceTypeT,
              template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSource,
              typename CreateFnT>
    auto make_source(std::string name, CreateFnT&& create_fn)
    {
        return construct_object<NodeTypeT<SourceTypeT>>(
            name,
            rxcpp::observable<>::create<SourceTypeT>(std::forward<CreateFnT>(create_fn)));
    }

    template <typename SourceTypeT, template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSource>
    auto make_source(std::string name, rxcpp::observable<SourceTypeT> obs)
    {
        return construct_object<NodeTypeT<SourceTypeT>>(name, obs);
    }

    template <typename SinkTypeT,
              template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSink,
              typename... ArgsT>
    auto make_sink(std::string name, ArgsT&&... ops)
    {
        return construct_object<NodeTypeT<SinkTypeT>>(name,
                                                      rxcpp::make_observer<SinkTypeT>(std::forward<ArgsT>(ops)...));
    }

    template <typename SinkTypeT, template <class> class NodeTypeT = node::RxSinkComponent, typename... ArgsT>
    auto make_sink_component(std::string name, ArgsT&&... ops)
    {
        return construct_object<NodeTypeT<SinkTypeT>>(name,
                                                      rxcpp::make_observer<SinkTypeT>(std::forward<ArgsT>(ops)...));
    }

    template <typename SinkTypeT,
              template <class, class, class = mrc::runnable::Context> class NodeTypeT = node::RxNode,
              typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops)
    {
        return construct_object<NodeTypeT<SinkTypeT, SinkTypeT>>(name, std::forward<ArgsT>(ops)...);
    }

    template <typename SinkTypeT,
              typename SourceTypeT,
              template <class, class, class = mrc::runnable::Context> class NodeTypeT = node::RxNode,
              typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops)
    {
        return construct_object<NodeTypeT<SinkTypeT, SourceTypeT>>(name, std::forward<ArgsT>(ops)...);
    }

    template <typename SinkTypeT,
              typename SourceTypeT,
              template <class, class> class NodeTypeT = node::RxNodeComponent,
              typename... ArgsT>
    auto make_node_component(std::string name, ArgsT&&... ops)
    {
        return construct_object<NodeTypeT<SinkTypeT, SourceTypeT>>(name, std::forward<ArgsT>(ops)...);
    }

    /**
     * Instantiate a segment module of `ModuleTypeT`, intialize it, and return it to the caller
     * @tparam ModuleTypeT Type of module to create
     * @param module_name Unique name of this instance of the module
     * @param config Configuration to pass to the module
     * @return Return a shared pointer to the new module, which is a derived class of SegmentModule
     */
    template <typename ModuleTypeT>
    std::shared_ptr<ModuleTypeT> make_module(std::string module_name, nlohmann::json config = {})
    {
        static_assert(std::is_base_of_v<modules::SegmentModule, ModuleTypeT>);

        auto module = std::make_shared<ModuleTypeT>(std::move(module_name), std::move(config));
        init_module(module);

        return std::move(module);
    }

    /**
     * Initialize a SegmentModule that was instantiated outside of the builder.
     * @param module Module to initialize
     */
    void init_module(std::shared_ptr<mrc::modules::SegmentModule> module);

    /**
     * Register an input port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param input_name Unique name of the input port
     * @param object shared pointer to type erased Object associated with 'input_name' on this module instance.
     */
    void register_module_input(std::string input_name, std::shared_ptr<segment::ObjectProperties> object);

    /**
     * Get the json configuration for the current module under configuration.
     * @return nlohmann::json object.
     */
    nlohmann::json get_current_module_config();

    /**
     * Register an output port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param output_name Unique name of the output port
     * @param object shared pointer to type erased Object associated with 'output_name' on this module instance.
     */
    void register_module_output(std::string output_name, std::shared_ptr<segment::ObjectProperties> object);

    std::shared_ptr<mrc::modules::SegmentModule> load_module_from_registry(const std::string& module_id,
                                                                           const std::string& registry_namespace,
                                                                           std::string module_name,
                                                                           nlohmann::json config = {});

    template <typename SourceNodeTypeT = std::nullptr_t,
              typename SinkNodeTypeT   = SourceNodeTypeT,
              MRCObjectPropRepr SourceTypeT,
              MRCObjectPropRepr SinkTypeT>
    void make_edge(SourceTypeT source, SinkTypeT sink);

    template <typename EdgeDataTypeT, MRCObjectPropRepr TapInputReprT, MRCObjectPropRepr TapOutputReprT>
    void make_edge_tap(std::string source, std::string sink, TapInputReprT tap_input, TapOutputReprT tap_output);

    template <typename ObjectT>
    void add_throughput_counter(std::shared_ptr<segment::Object<ObjectT>> segment_object)
    {
        auto runnable = std::dynamic_pointer_cast<Runnable<ObjectT>>(segment_object);
        CHECK(runnable);
        CHECK(segment_object->is_source());
        using source_type_t = typename ObjectT::source_type_t;
        auto counter        = m_backend.make_throughput_counter(runnable->name());
        runnable->object().add_epilogue_tap([counter](const source_type_t& data) {
            counter(1);
        });
    }

    template <typename ObjectT, typename CallableT>
    void add_throughput_counter(std::shared_ptr<segment::Object<ObjectT>> segment_object, CallableT&& callable)
    {
        auto runnable = std::dynamic_pointer_cast<Runnable<ObjectT>>(segment_object);
        CHECK(runnable);
        CHECK(segment_object->is_source());
        using source_type_t = typename ObjectT::source_type_t;
        using tick_fn_t     = std::function<std::int64_t(const source_type_t&)>;
        tick_fn_t tick_fn   = callable;
        auto counter        = m_backend.make_throughput_counter(runnable->name());
        runnable->object().add_epilogue_tap([counter, tick_fn](const source_type_t& data) {
            counter(tick_fn(data));
        });
    }

  private:
    using sp_segment_module_t = std::shared_ptr<mrc::modules::SegmentModule>;

    std::string m_namespace_prefix;
    std::vector<std::string> m_namespace_stack{};
    std::vector<sp_segment_module_t> m_module_stack{};

    internal::segment::IBuilder& m_backend;

    void ns_push(sp_segment_module_t module);

    void ns_pop();

    template <MRCObjectPropRepr ObjectReprT>
    ObjectProperties& to_object_properties(ObjectReprT& repr);

    friend Definition;
};

template <MRCObjectPropRepr ObjectReprT>
ObjectProperties& Builder::to_object_properties(ObjectReprT& repr)
{
    ObjectProperties* object_properties_ptr{nullptr};
    if constexpr (is_shared_ptr_v<ObjectReprT>)
    {
        // SP to Object
        if constexpr (is_object_v<typename ObjectReprT::element_type>)
        {
            auto object_properties_ptr_props_ptr = std::dynamic_pointer_cast<ObjectProperties>(repr);
            object_properties_ptr                = std::addressof(*object_properties_ptr_props_ptr);
        }
        // SP to ObjectProperties
        else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ObjectReprT>>,
                                          std::shared_ptr<ObjectProperties>>)
        {
            object_properties_ptr = std::addressof(*repr);
        }
    }
    // Object
    else if constexpr (is_object_v<ObjectReprT>)
    {
        object_properties_ptr = std::addressof(dynamic_cast<ObjectProperties&>(repr));
    }
    // ObjectProperties
    else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ObjectReprT>>, ObjectProperties>)
    {
        object_properties_ptr = std::addressof(repr);
    }
    // String-like lookup
    else
    {
        object_properties_ptr = std::addressof(m_backend.find_object(repr));
    }

    DCHECK(object_properties_ptr != nullptr) << "If this fails, something is wrong with the concept definition";

    return *object_properties_ptr;
}

template <typename SourceNodeTypeT, typename SinkNodeTypeT, MRCObjectPropRepr SourceTypeT, MRCObjectPropRepr SinkTypeT>
void Builder::make_edge(SourceTypeT source, SinkTypeT sink)
{
    DVLOG(10) << "forming segment edge between two segment objects";
    using source_shared_ptr_type_t = ObjectSharedPtrType<SourceTypeT>::type_t::source_type_t;  // Might be nullptr_t
    using sink_shared_ptr_type_t   = ObjectSharedPtrType<SinkTypeT>::type_t::sink_type_t;      // Might be nullptr_t

    auto& source_object = to_object_properties(source);
    auto& sink_object   = to_object_properties(sink);

    using deduced_source_type_t = FirstNonNullType<SourceNodeTypeT,                  // Explicit type hint
                                                   source_shared_ptr_type_t,         // Deduced type (if possible)
                                                   SinkNodeTypeT,                    // Fallback to Sink explicit hint
                                                   sink_shared_ptr_type_t>::type_t;  // Fallback to sink
                                                                                     // deduced type
    using deduced_sink_type_t =
        FirstNonNullType<SinkNodeTypeT, sink_shared_ptr_type_t, SourceNodeTypeT, source_shared_ptr_type_t>::type_t;

    VLOG(2) << "Deduced source type: " << mrc::boost_type_name<deduced_source_type_t>() << std::endl;
    VLOG(2) << "Deduced sink type: " << mrc::boost_type_name<deduced_sink_type_t>() << std::endl;

    if (source_object.is_writable_acceptor() && sink_object.is_writable_provider())
    {
        mrc::make_edge(source_object.template writable_acceptor_typed<deduced_source_type_t>(),
                       sink_object.template writable_provider_typed<deduced_sink_type_t>());
        return;
    }

    if (source_object.is_readable_provider() && sink_object.is_readable_acceptor())
    {
        mrc::make_edge(source_object.template readable_provider_typed<deduced_source_type_t>(),
                       sink_object.template readable_acceptor_typed<deduced_sink_type_t>());
        return;
    }

    LOG(ERROR) << "Incompatible node types";
}

template <typename EdgeDataTypeT, MRCObjectPropRepr TapInputReprT, MRCObjectPropRepr TapOutputReprT>
void Builder::make_edge_tap(const std::string source,
                            const std::string sink,
                            TapInputReprT tap_input,
                            TapOutputReprT tap_output)
{
    auto& source_object = to_object_properties(source);
    auto& sink_object   = to_object_properties(sink);

    CHECK(source_object.is_source()) << "Source object is not a source";
    CHECK(sink_object.is_sink()) << "Sink object is not a sink";

    // 'standard' Source->Sink connection
    if (source_object.is_writable_acceptor())
    {
        if (sink_object.is_writable_provider())
        {
            CHECK(tap_input->is_writable_provider()) << "Tap input must be of type WritableProvider";
            CHECK(tap_output->is_writable_acceptor()) << "Tap output must be WritableAcceptor";

            // Cast our object into something we can insert as a tap.
            auto& tap_input_writable_provider  = tap_input->template writable_provider_typed<EdgeDataTypeT>();
            auto& tap_output_writable_acceptor = tap_output->template writable_acceptor_typed<EdgeDataTypeT>();

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
            auto& source_acceptor    = source_object.writable_acceptor_typed<EdgeDataTypeT>();
            auto& source_edge_holder = dynamic_cast<edge::EdgeHolder<EdgeDataTypeT>&>(source_acceptor);

            auto& sink_provider = sink_object.writable_provider_typed<EdgeDataTypeT>();

            // Check that an active connection exists before trying to splice.
            CHECK(source_edge_holder.check_active_connection(false)) << "No active connection to tap";

            // Copy the Acceptor's edge handle and release it from the Acceptor
            // Make sure we hold the edge handle until the new edge to the tap has been formed.
            // TODO(Devin): Can we double check that the edge handle from the source matches the one from the sink?
            auto edge_handle = source_edge_holder.get_connected_edge();
            source_edge_holder.release_edge_connection();

            edge::EdgeBuilder::make_edge_writable(source_acceptor, tap_input_writable_provider);
            edge::EdgeBuilder::make_edge_writable(tap_output_writable_acceptor, sink_provider);

            return;
        }
    }

    throw std::runtime_error("Attempt to splice unsupported edge types");
}

template <typename ObjectT>
std::shared_ptr<Object<ObjectT>> Builder::make_object(std::string name, std::unique_ptr<ObjectT> node)
{
    // Note: name should have any prefix modifications done prior to getting here.
    if (m_backend.has_object(name))
    {
        LOG(ERROR) << "A Object named " << name << " is already registered";
        throw exceptions::MrcRuntimeError("duplicate name detected - name owned by a node");
    }

    std::shared_ptr<Object<ObjectT>> segment_object{nullptr};

    if constexpr (std::is_base_of_v<runnable::Runnable, ObjectT>)
    {
        auto segment_name = m_backend.name() + "/" + name;
        auto segment_node = std::make_shared<Runnable<ObjectT>>(segment_name, std::move(node));

        std::cerr << "Adding Object: " << name << std::endl << std::flush;
        m_backend.add_runnable(name, segment_node);
        m_backend.add_object(name, segment_node);
        segment_object = segment_node;
    }
    else
    {
        auto segment_node = std::make_shared<Component<ObjectT>>(std::move(node));
        std::cerr << "Adding Object: " << name << std::endl << std::flush;
        m_backend.add_object(name, segment_node);
        segment_object = segment_node;
    }

    CHECK(segment_object);
    return segment_object;
}

template <typename T>
std::shared_ptr<Object<node::RxSinkBase<T>>> Builder::get_egress(std::string name)
{
    auto base = m_backend.get_egress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Egress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<Object<node::RxSinkBase<T>>>(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

template <typename T>
std::shared_ptr<Object<node::RxSourceBase<T>>> Builder::get_ingress(std::string name)
{
    auto base = m_backend.get_ingress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Ingress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<Object<node::RxSourceBase<T>>>(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Ingress port type mismatch: " + name);
    }

    return port;
}

}  // namespace mrc::segment
