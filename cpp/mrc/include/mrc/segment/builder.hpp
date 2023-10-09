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

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"    // IWYU pragma: keep
#include "mrc/node/source_properties.hpp"  // IWYU pragma: keep
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/runnable.hpp"  // IWYU pragma: keep
#include "mrc/segment/component.hpp"  // IWYU pragma: keep
#include "mrc/segment/concepts/object_traits.hpp"
#include "mrc/segment/object.hpp"    // IWYU pragma: keep
#include "mrc/segment/runnable.hpp"  // IWYU pragma: keep
#include "mrc/type_traits.hpp"
#include "mrc/utils/macros.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/hana/core/when.hpp>
#include <boost/hana/if.hpp>
#include <boost/hana/type.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc {
struct WatcherInterface;
}  // namespace mrc

namespace mrc::modules {
class SegmentModule;
}  // namespace mrc::modules

namespace mrc::node {
template <typename T>
class RxSinkBase;
}  // namespace mrc::node

namespace mrc::node {
template <typename T>
class RxSourceBase;
}  // namespace mrc::node

namespace mrc::segment {
class Definition;
}  // namespace mrc::segment

namespace {
namespace hana = boost::hana;

template <typename T>
auto has_source_add_watcher =
    hana::is_valid([](auto&& thing) -> decltype(std::forward<decltype(thing)>(thing).source_add_watcher(
                                        std::declval<std::shared_ptr<mrc::WatcherInterface>>())) {});

template <typename T>
auto has_sink_add_watcher =
    hana::is_valid([](auto&& thing) -> decltype(std::forward<decltype(thing)>(thing).sink_add_watcher(
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

class IBuilder
{
  public:
    IBuilder()          = default;
    virtual ~IBuilder() = default;

    DELETE_COPYABILITY(IBuilder);

    virtual const std::string& name() const = 0;

    /**
     * @brief Takes either a local or global object name and returns the global name and local name separately. Global
     * names contain '/<SegmentName>/<m_namespace_prefix>/<name>' (leading '/') where local names are
     * '<m_namespace_prefix>/<name>' (no leading '/')
     *
     * @param name Name to normalize
     * @param ignore_namespace Whether or not to ignore the '<m_namespace_prefix>' portion. Useful for ports.
     * @return std::tuple<std::string, std::string> Global name, Local name
     */
    virtual std::tuple<std::string, std::string> normalize_name(const std::string& name,
                                                                bool ignore_namespace = false) const = 0;

    virtual std::shared_ptr<ObjectProperties> get_ingress(std::string name, std::type_index type_index) = 0;

    virtual std::shared_ptr<ObjectProperties> get_egress(std::string name, std::type_index type_index) = 0;

    /**
     * Initialize a SegmentModule that was instantiated outside of the builder.
     * @param module Module to initialize
     */
    virtual void init_module(std::shared_ptr<mrc::modules::SegmentModule> smodule) = 0;

    /**
     * Register an input port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param input_name Unique name of the input port
     * @param object shared pointer to type erased Object associated with 'input_name' on this module instance.
     */
    virtual void register_module_input(std::string input_name, std::shared_ptr<ObjectProperties> object) = 0;

    /**
     * Get the json configuration for the current module under configuration.
     * @return nlohmann::json object.
     */
    virtual nlohmann::json get_current_module_config() = 0;

    /**
     * Register an output port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param output_name Unique name of the output port
     * @param object shared pointer to type erased Object associated with 'output_name' on this module instance.
     */
    virtual void register_module_output(std::string output_name, std::shared_ptr<ObjectProperties> object) = 0;

    /**
     * Load an existing, registered module, initialize it, and return it to the caller
     * @param module_id Unique ID of the module to load
     * @param registry_namespace Namespace where the module id is registered
     * @param module_name Unique name of this instance of the module
     * @param config Configuration to pass to the module
     * @return Return a shared pointer to the new module, which is a derived class of SegmentModule
     */
    virtual std::shared_ptr<mrc::modules::SegmentModule> load_module_from_registry(const std::string& module_id,
                                                                                   const std::string& registry_namespace,
                                                                                   std::string module_name,
                                                                                   nlohmann::json config = {}) = 0;

    template <typename ObjectT, typename... ArgsT>
    std::shared_ptr<Object<ObjectT>> construct_object(std::string name, ArgsT&&... args);

    template <typename ObjectT>
    std::shared_ptr<Object<ObjectT>> make_object(std::string name, std::unique_ptr<ObjectT> node);

    template <typename T>
    std::shared_ptr<Object<node::RxSinkBase<T>>> get_egress(std::string name);

    template <typename T>
    std::shared_ptr<Object<node::RxSourceBase<T>>> get_ingress(std::string name);

    /**
     * Create a source node using the provided name and function, the function is lifted to an observable
     * @tparam SourceTypeT the type of data produced by the source
     * @tparam NodeTypeT the type of node to use as the source
     * @tparam CreateFnT the type of function used to create the source
     * @param name the name of the source
     * @param create_fn the function that will be lifted into a Observable and used to create the source
     * @return the created source
     */
    template <typename SourceTypeT,
              template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSource,
              typename CreateFnT>
    auto make_source(std::string name, CreateFnT&& create_fn);

    /**
     * Create a source node using the provided name and observable
     * @tparam SourceTypeT The type of elements emitted by the source node
     * @tparam NodeTypeT The type of node to be created, with default set to node::RxSource
     * @param name The name of the source node
     * @param obs The observable emitting elements of type SourceTypeT
     * @return An object of type NodeTypeT<SourceTypeT>
     */
    template <typename SourceTypeT, template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSource>
    auto make_source(std::string name, rxcpp::observable<SourceTypeT> obs);

    /**
     * Creates a new instance of the sink component of type `NodeTypeT` with name `name` and observer `ops`.
     * @tparam SinkTypeT Type of the data received by the sink component.
     * @tparam NodeTypeT Type of the node to be constructed.
     * @tparam ArgsT Types of arguments passed to the observer.
     * @param name The name of the sink component.
     * @param ops The observer for the sink component.
     * @return The constructed sink as an Object.
     */
    template <typename SinkTypeT,
              template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSink,
              typename... ArgsT>
    auto make_sink(std::string name, ArgsT&&... ops);

    /**
     * Creates a sink component and returns it as an object
     * @tparam SinkTypeT the type of data that the sink component will receive
     * @tparam NodeTypeT the type of the node component to be constructed
     * @tparam ArgsT the type of the arguments for constructing the node component
     * @param name the name of the sink component
     * @param ops the arguments for constructing the node component
     * @return The constructed sink component as an Object
     */
    template <typename SinkTypeT, template <class> class NodeTypeT = node::RxSinkComponent, typename... ArgsT>
    auto make_sink_component(std::string name, ArgsT&&... ops);

    /**
     * Creates a new instance of the specified node type, with the given name and arguments.
     * @tparam SinkTypeT The type of the sink to be created.
     * @tparam SourceTypeT The type of the source to be created.
     * @tparam NodeTypeT The type of the node to be created.
     * @tparam ArgsT Variadic parameter pack, representing the arguments to be passed to the constructor of the node.
     * @param name The name of the node to be created.
     * @param ops The arguments to be passed to the constructor of the node.
     * @return The newly created node.
     */
    template <typename SinkTypeT,
              template <class, class, class = mrc::runnable::Context> class NodeTypeT = node::RxNode,
              typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops);

    /**
     * Constructs a node object of type NodeTypeT<SinkTypeT, SourceTypeT>, initialized with arguments ops
     * @tparam SinkTypeT The type of the sink
     * @tparam SourceTypeT The type of the source
     * @tparam NodeTypeT The type of the node, defaults to mrc::node::RxNode
     * @tparam ArgsT The types of the arguments passed to the node constructor
     * @param name The name of the node
     * @param ops The arguments passed to the node constructor
     * @return An object of type NodeTypeT<SinkTypeT, SourceTypeT> constructed with the provided arguments
     */
    template <typename SinkTypeT,
              typename SourceTypeT,
              template <class, class, class = mrc::runnable::Context> class NodeTypeT = node::RxNode,
              typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops);

    /**
     * Creates and returns an instance of a node component with the specified type, name and arguments.
     * @tparam SinkTypeT The sink type of the node component to be created.
     * @tparam SourceTypeT The source type of the node component to be created.
     * @tparam NodeTypeT The type of the node component to be created.
     * @tparam ArgsT Variadic template argument for the node component's constructor.
     * @param name The name of the node component to be created.
     * @param ops Variadic argument for the node component's constructor.
     * @return An instance of the created node component.
     */
    template <typename SinkTypeT,
              typename SourceTypeT,
              template <class, class> class NodeTypeT = node::RxNodeComponent,
              typename... ArgsT>
    auto make_node_component(std::string name, ArgsT&&... ops);

    /**
     * Instantiate a segment module of `ModuleTypeT`, intialize it, and return it to the caller
     * @tparam ModuleTypeT Type of module to create
     * @param module_name Unique name of this instance of the module
     * @param config Configuration to pass to the module
     * @return Return a shared pointer to the new module, which is a derived class of SegmentModule
     */
    template <typename ModuleTypeT>
    std::shared_ptr<ModuleTypeT> make_module(std::string module_name, nlohmann::json config = {});

    /**
     * Create an edge between two things that are convertible to ObjectProperties
     * @tparam SourceNodeTypeT Type hint for the source node -- optional -- this will be used if the type of the source
     * object cannot be directly determined.
     * @tparam SinkNodeTypeT Type hint for the sink node -- optional -- this will be used if the type of the sink object
     * cannot be directly determined.
     * @tparam SourceObjectT Concept conforming type of the source object
     * @tparam SinkObjectT Concept conforming type of the sink object
     * @param source Edge source
     * @param sink Edge Sink
     */
    template <typename SourceNodeTypeT = void,
              typename SinkNodeTypeT   = SourceNodeTypeT,
              MRCObjectProxy SourceObjectT,
              MRCObjectProxy SinkObjectT>
    void make_edge(SourceObjectT source, SinkObjectT sink);

    /**
     *
     * @tparam EdgeDataTypeT
     * @tparam SourceObjectT Data type that can be resolved to an ObjectProperties, representing the source
     * @tparam SinkObjectT Data type that can be resolved to an ObjectProperties, representing the sink
     * @tparam SpliceInputObjectT Data type that can be resolved to an ObjectProperties, representing the splice's input
     * @tparam SpliceOutputObjectT Data type that can be resolved to an ObjectProperties, representing the splice's
     * output
     * @param source Existing, connected, edge source
     * @param sink Existing, connected, edge sink
     * @param splice_input Existing, unconnected, splice input
     * @param splice_output Existing, unconnected, splice output
     */
    template <typename EdgeDataTypeT,
              MRCObjectProxy SourceObjectT,
              MRCObjectProxy SinkObjectT,
              MRCObjectProxy SpliceInputObjectT,
              MRCObjectProxy SpliceOutputObjectT>
    void splice_edge(SourceObjectT source,
                     SinkObjectT sink,
                     SpliceInputObjectT splice_input,
                     SpliceOutputObjectT splice_output);

    template <typename ObjectT>
    void add_throughput_counter(std::shared_ptr<Object<ObjectT>> segment_object);

    template <typename ObjectT, typename CallableT>
    void add_throughput_counter(std::shared_ptr<Object<ObjectT>> segment_object, CallableT&& callable);

  private:
    virtual ObjectProperties& find_object(const std::string& name)                             = 0;
    virtual void add_object(const std::string& name, std::shared_ptr<ObjectProperties> object) = 0;
    virtual std::shared_ptr<IngressPortBase> get_ingress_base(const std::string& name)         = 0;
    virtual std::shared_ptr<EgressPortBase> get_egress_base(const std::string& name)           = 0;
    virtual std::function<void(std::int64_t)> make_throughput_counter(const std::string& name) = 0;

    template <MRCObjectProxy ObjectReprT>
    ObjectProperties& to_object_properties(ObjectReprT& repr);
};

template <typename ObjectT, typename... ArgsT>
std::shared_ptr<Object<ObjectT>> IBuilder::construct_object(std::string name, ArgsT&&... args)
{
    auto uptr = std::make_unique<ObjectT>(name, std::forward<ArgsT>(args)...);

    return make_object(std::move(name), std::move(uptr));
}

template <typename ObjectT>
std::shared_ptr<Object<ObjectT>> IBuilder::make_object(std::string name, std::unique_ptr<ObjectT> node)
{
    std::shared_ptr<Object<ObjectT>> segment_object{nullptr};

    if constexpr (std::is_base_of_v<runnable::Runnable, ObjectT>)
    {
        segment_object = std::make_shared<Runnable<ObjectT>>(name, std::move(node));
        this->add_object(std::move(name), segment_object);
    }
    else
    {
        segment_object = std::make_shared<Component<ObjectT>>(name, std::move(node));
        this->add_object(std::move(name), segment_object);
    }

    CHECK(segment_object);

    // Now that we have been added, set the stats watchers using the object name
    ::add_stats_watcher_if_rx_source(segment_object->object(), segment_object->name());
    ::add_stats_watcher_if_rx_sink(segment_object->object(), segment_object->name());

    return segment_object;
}

template <typename SourceTypeT, template <class, class = mrc::runnable::Context> class NodeTypeT, typename CreateFnT>
auto IBuilder::make_source(std::string name, CreateFnT&& create_fn)
{
    return construct_object<NodeTypeT<SourceTypeT>>(
        name,
        rxcpp::observable<>::create<SourceTypeT>(std::forward<CreateFnT>(create_fn)));
}

template <typename SourceTypeT, template <class, class = mrc::runnable::Context> class NodeTypeT>
auto IBuilder::make_source(std::string name, rxcpp::observable<SourceTypeT> obs)
{
    return construct_object<NodeTypeT<SourceTypeT>>(name, obs);
}

template <typename SinkTypeT, template <class, class = mrc::runnable::Context> class NodeTypeT, typename... ArgsT>
auto IBuilder::make_sink(std::string name, ArgsT&&... ops)
{
    return construct_object<NodeTypeT<SinkTypeT>>(name, rxcpp::make_observer<SinkTypeT>(std::forward<ArgsT>(ops)...));
}

template <typename SinkTypeT, template <class> class NodeTypeT, typename... ArgsT>
auto IBuilder::make_sink_component(std::string name, ArgsT&&... ops)
{
    return construct_object<NodeTypeT<SinkTypeT>>(name, rxcpp::make_observer<SinkTypeT>(std::forward<ArgsT>(ops)...));
}

template <typename SinkTypeT, template <class, class, class = mrc::runnable::Context> class NodeTypeT, typename... ArgsT>
auto IBuilder::make_node(std::string name, ArgsT&&... ops)
{
    return construct_object<NodeTypeT<SinkTypeT, SinkTypeT>>(name, std::forward<ArgsT>(ops)...);
}

template <typename SinkTypeT,
          typename SourceTypeT,
          template <class, class, class = mrc::runnable::Context>
          class NodeTypeT,
          typename... ArgsT>
auto IBuilder::make_node(std::string name, ArgsT&&... ops)
{
    return construct_object<NodeTypeT<SinkTypeT, SourceTypeT>>(name, std::forward<ArgsT>(ops)...);
}

template <typename SinkTypeT, typename SourceTypeT, template <class, class> class NodeTypeT, typename... ArgsT>
auto IBuilder::make_node_component(std::string name, ArgsT&&... ops)
{
    return construct_object<NodeTypeT<SinkTypeT, SourceTypeT>>(name, std::forward<ArgsT>(ops)...);
}

template <typename ModuleTypeT>
std::shared_ptr<ModuleTypeT> IBuilder::make_module(std::string module_name, nlohmann::json config)
{
    static_assert(std::is_base_of_v<modules::SegmentModule, ModuleTypeT>);

    auto smodule = std::make_shared<ModuleTypeT>(std::move(module_name), std::move(config));
    init_module(smodule);

    return std::move(smodule);
}

template <typename SourceNodeTypeT, typename SinkNodeTypeT, MRCObjectProxy SourceObjectT, MRCObjectProxy SinkObjectT>
void IBuilder::make_edge(SourceObjectT source, SinkObjectT sink)

{
    DVLOG(10) << "forming edge between two segment objects";
    using source_sp_type_t = typename mrc_object_sptr_type_t<SourceObjectT>::source_type_t;  // Might be void
    using sink_sp_type_t   = typename mrc_object_sptr_type_t<SinkObjectT>::sink_type_t;      // Might be void

    auto& source_object = to_object_properties(source);
    auto& sink_object   = to_object_properties(sink);

    // If we can determine the type from the actual object, use that, then fall back to hints or defaults.
    using deduced_source_type_t = first_non_void_type_t<source_sp_type_t,  // Deduced type (if possible)
                                                        SourceNodeTypeT,   // Explicit type hint
                                                        sink_sp_type_t,    // Fallback to Sink deduced type
                                                        SinkNodeTypeT>;    // Fallback to Sink explicit hint
    using deduced_sink_type_t   = first_non_void_type_t<sink_sp_type_t,    // Deduced type (if possible)
                                                      SinkNodeTypeT,     // Explicit type hint
                                                      source_sp_type_t,  // Fallback to Source deduced type
                                                      SourceNodeTypeT>;  // Fallback to Source explicit hint

    VLOG(2) << "Deduced source type: " << mrc::type_name<deduced_source_type_t>() << std::endl;
    VLOG(2) << "Deduced sink type: " << mrc::type_name<deduced_sink_type_t>() << std::endl;

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

template <typename EdgeDataTypeT,
          MRCObjectProxy SourceObjectT,
          MRCObjectProxy SinkObjectT,
          MRCObjectProxy SpliceInputObjectT,
          MRCObjectProxy SpliceOutputObjectT>
void IBuilder::splice_edge(SourceObjectT source,
                           SinkObjectT sink,
                           SpliceInputObjectT splice_input,
                           SpliceOutputObjectT splice_output)

{
    auto& source_object = to_object_properties(source);
    auto& sink_object   = to_object_properties(sink);

    auto& splice_input_object  = to_object_properties(splice_input);
    auto& splice_output_object = to_object_properties(splice_output);

    CHECK(source_object.is_source()) << "Source object is not a source";
    CHECK(sink_object.is_sink()) << "Sink object is not a sink";

    // TODO(Devin): this is slightly more constrained that it needs to be. We don't actually need to know the
    // type of a provider, but because of the way type testing is done on the edg ebuilder, its a bit of a pain
    // to pass in an untyped Provider. We can fix this later.
    if (source_object.is_writable_acceptor())
    {
        if (sink_object.is_writable_provider())
        {
            CHECK(splice_input_object.is_writable_provider()) << "Splice input must be of type WritableProvider";
            CHECK(splice_output_object.is_writable_acceptor()) << "Splice output must be WritableAcceptor";

            // Cast our object into something we can insert as a splice.
            auto& splice_writable_provider = splice_input_object.template writable_provider_typed<EdgeDataTypeT>();
            auto& splice_writable_acceptor = splice_output_object.template writable_acceptor_typed<EdgeDataTypeT>();

            auto& writable_acceptor = source_object.template writable_acceptor_typed<EdgeDataTypeT>();
            auto& writable_provider = sink_object.template writable_provider_typed<EdgeDataTypeT>();

            edge::EdgeBuilder::splice_edge<EdgeDataTypeT>(writable_acceptor,
                                                          writable_provider,
                                                          splice_writable_provider,
                                                          splice_writable_acceptor);

            return;
        }
    }
    else if (source_object.is_readable_provider())
    {
        if (sink_object.is_readable_acceptor())
        {
            CHECK(splice_input_object.is_readable_acceptor()) << "Splice input must be of type ReadableAcceptor";
            CHECK(splice_output_object.is_readable_provider()) << "Splice output must be ReadableProvider";

            // Cast our object into something we can insert as a splice.
            auto& splice_readable_acceptor = splice_input_object.template readable_acceptor_typed<EdgeDataTypeT>();
            auto& splice_readable_provider = splice_output_object.template readable_provider_typed<EdgeDataTypeT>();

            auto& readable_provider = source_object.template readable_provider_typed<EdgeDataTypeT>();
            auto& readable_acceptor = sink_object.template readable_acceptor_typed<EdgeDataTypeT>();

            edge::EdgeBuilder::splice_edge<EdgeDataTypeT>(readable_provider,
                                                          readable_acceptor,
                                                          splice_readable_acceptor,
                                                          splice_readable_provider);

            return;
        }
    }

    throw std::runtime_error("Attempt to splice unsupported edge types");
}

template <typename T>
std::shared_ptr<Object<node::RxSinkBase<T>>> IBuilder::get_egress(std::string name)
{
    auto base = this->get_egress_base(name);
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
std::shared_ptr<Object<node::RxSourceBase<T>>> IBuilder::get_ingress(std::string name)
{
    auto base = this->get_ingress_base(name);
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

template <typename ObjectT>
void IBuilder::add_throughput_counter(std::shared_ptr<Object<ObjectT>> segment_object)
{
    auto runnable = std::dynamic_pointer_cast<Runnable<ObjectT>>(segment_object);
    CHECK(runnable);
    CHECK(segment_object->is_source());
    using source_type_t = typename ObjectT::source_type_t;
    auto counter        = this->make_throughput_counter(runnable->name());
    runnable->object().add_epilogue_tap([counter](const source_type_t& data) {
        counter(1);
    });
}

template <typename ObjectT, typename CallableT>
void IBuilder::add_throughput_counter(std::shared_ptr<Object<ObjectT>> segment_object, CallableT&& callable)
{
    auto runnable = std::dynamic_pointer_cast<Runnable<ObjectT>>(segment_object);
    CHECK(runnable);
    CHECK(segment_object->is_source());
    using source_type_t = typename ObjectT::source_type_t;
    using tick_fn_t     = std::function<std::int64_t(const source_type_t&)>;
    tick_fn_t tick_fn   = callable;
    auto counter        = this->make_throughput_counter(runnable->name());
    runnable->object().add_epilogue_tap([counter, tick_fn](const source_type_t& data) {
        counter(tick_fn(data));
    });
}

/* Private Member Functions */
template <MRCObjectProxy ObjectReprT>
ObjectProperties& IBuilder::to_object_properties(ObjectReprT& repr)
{
    ObjectProperties* object_properties_ptr{nullptr};
    if constexpr (is_shared_ptr_v<ObjectReprT>)
    {
        // SP to Object
        if constexpr (MRCObjectSharedPtr<ObjectReprT>)
        {
            auto object_properties_ptr_props_ptr = std::dynamic_pointer_cast<ObjectProperties>(repr);
            object_properties_ptr                = std::addressof(*object_properties_ptr_props_ptr);
        }
        // SP to ObjectProperties
        else if constexpr (MRCObjPropSharedPtr<ObjectReprT>)
        {
            object_properties_ptr = std::addressof(*repr);
        }
        {
            object_properties_ptr = std::addressof(*repr);
        }
    }
    // Object
    else if constexpr (MRCObject<ObjectReprT>)
    {
        object_properties_ptr = std::addressof(dynamic_cast<ObjectProperties&>(repr));
    }
    // ObjectProperties
    else if constexpr (MRCObjProp<ObjectReprT>)
    {
        object_properties_ptr = std::addressof(repr);
    }
    // String-like lookup
    else
    {
        object_properties_ptr = std::addressof(this->find_object(repr));
    }

    CHECK(object_properties_ptr != nullptr) << "If this fails, something is wrong with the concept definition";

    return *object_properties_ptr;
}

// For backwards compatibility, make a type alias to `Builder`
using Builder = IBuilder;  // NOLINT(readability-identifier-naming)

}  // namespace mrc::segment
