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
#include "mrc/core/watcher.hpp"
#include "mrc/engine/segment/ibuilder.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/sink_properties.hpp"    // IWYU pragma: keep
#include "mrc/node/source_properties.hpp"  // IWYU pragma: keep
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/launchable.hpp"   // IWYU pragma: keep
#include "mrc/runnable/runnable.hpp"     // IWYU pragma: keep
#include "mrc/segment/component.hpp"     // IWYU pragma: keep
#include "mrc/segment/egress_port.hpp"   // IWYU pragma: keep
#include "mrc/segment/forward.hpp"       // IWYU pragma: keep
#include "mrc/segment/ingress_port.hpp"  // IWYU pragma: keep
#include "mrc/segment/object.hpp"        // IWYU pragma: keep
#include "mrc/segment/runnable.hpp"      // IWYU pragma: keep
#include "mrc/utils/macros.hpp"

#include <boost/hana.hpp>  // IWYU pragma: keep
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

// IWYU pragma: no_include <boost/hana/fwd/core/when.hpp>
// IWYU pragma: no_include <boost/hana/fwd/if.hpp>
// IWYU pragma: no_include <boost/hana/fwd/type.hpp>

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

class Builder final
{
    Builder(internal::segment::IBuilder& backend) : m_backend(backend) {}

  public:
    DELETE_COPYABILITY(Builder);
    DELETE_MOVEABILITY(Builder);

    std::shared_ptr<ObjectProperties> get_ingress(std::string name, std::type_index type_index);
    std::shared_ptr<ObjectProperties> get_egress(std::string name, std::type_index type_index);

    template <typename T>
    std::shared_ptr<Object<node::SinkProperties<T>>> get_egress(std::string name);

    template <typename T>
    std::shared_ptr<Object<node::SourceProperties<T>>> get_ingress(std::string name);

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
            name, rxcpp::observable<>::create<SourceTypeT>(std::forward<CreateFnT>(create_fn)));
    }

    template <typename SinkTypeT,
              template <class, class = mrc::runnable::Context> class NodeTypeT = node::RxSink,
              typename... ArgsT>
    auto make_sink(std::string name, ArgsT&&... ops)
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

    template <typename SourceNodeTypeT, typename SinkNodeTypeT>
    void make_edge(std::shared_ptr<Object<SourceNodeTypeT>> source, std::shared_ptr<Object<SinkNodeTypeT>> sink)
    {
        DVLOG(10) << "forming segment edge between two segment objects";
        node::make_edge(source->object(), sink->object());
    }

    template <typename InputT, typename SinkNodeTypeT>
    void make_edge(node::SourceProperties<InputT>& source, std::shared_ptr<Object<SinkNodeTypeT>> sink)
    {
        DVLOG(10) << "forming segment edge from node source to segment sink";
        node::make_edge(source, sink->object());
    }

    template <typename SourceNodeTypeT, typename OutputT>
    void make_edge(std::shared_ptr<Object<SourceNodeTypeT>>& source, node::SinkProperties<OutputT>& sink)
    {
        DVLOG(10) << "forming segment edge between a segment source and node sink";
        node::make_edge(source->object(), sink);
    }

    template <typename InputT, typename OutputT>
    void make_edge(node::SourceProperties<InputT>& source, node::SinkProperties<OutputT>& sink)
    {
        DVLOG(10) << "forming segment edge between two node objects";
        node::make_edge(source, sink);
    }

    /**
     * Given a typed source and a typeless sink, attempt to construct an edge between them -- assumes that source and
     * sink types are convertible.
     *
     * @tparam InputT
     * @param source
     * @param sink
     */
    template <typename InputT>
    void make_edge(node::SourceProperties<InputT>& source, std::shared_ptr<segment::ObjectProperties> sink)
    {
        DVLOG(10) << "forming segment edge between two node objects";
        node::make_edge(source, sink->template sink_typed<InputT>());
    }

    /**
     * Partial dynamic edge construction:
     *
     * Create edge using a fully constructed Object and a type erased Object
     *  We extract the underlying node object (Likely an RxNode) and call make_edge with it and the type erased
     *  object. This works via a cascaded type extraction process.
     * @tparam SourceNodeTypeT
     * @param source Fully typed, wrapped, object
     * @param sink Type erased object -- assumed to be convertible to source type
     */
    template <typename SourceNodeTypeT>
    void make_edge(std::shared_ptr<Object<SourceNodeTypeT>>& source, std::shared_ptr<segment::ObjectProperties> sink)
    {
        DVLOG(10) << "forming segment edge between a segment source and typeless Object";
        this->make_edge(source->object(), sink);
    }

    /**
     * Given a typeless source and a typed sink, attempt to construct an edge between them -- assumes that
     * source and sink type's are convertible.
     *
     * @tparam OutputT
     * @param source
     * @param sink
     */
    template <typename OutputT>
    void make_edge(std::shared_ptr<segment::ObjectProperties> source, node::SinkProperties<OutputT>& sink)
    {
        DVLOG(10) << "forming segment edge between two node objects";
        node::make_edge(source->template source_typed<OutputT>(), sink);
    }

    /**
     * Partial dynamic edge construction:
     *
     * Create edge using a fully constructed Object and a type erased Object
     *  We extract the underlying node object (Likely an RxNode) and call make_edge with it and the type erased
     *  object. This works via a cascaded type extraction process.
     * @tparam SinkNodeTypeT
     * @param source Fully typed, wrapped, object
     * @param sink Fully typed, wrapped, object
     */
    template <typename SinkNodeTypeT>
    void make_edge(std::shared_ptr<segment::ObjectProperties> source, std::shared_ptr<Object<SinkNodeTypeT>>& sink)
    {
        DVLOG(10) << "forming segment edge between a typeless object and a segment sink";
        this->make_edge(source, sink->object());
    }

    template <typename SourceNodeTypeT, typename SinkNodeTypeT = SourceNodeTypeT>
    void make_dynamic_edge(const std::string& source_name, const std::string& sink_name)
    {
        auto& source_obj = m_backend.find_object(source_name);
        auto& sink_obj   = m_backend.find_object(sink_name);
        node::make_edge(source_obj.source_typed<SourceNodeTypeT>(), sink_obj.sink_typed<SinkNodeTypeT>());
    }

    template <typename SourceNodeTypeT, typename SinkNodeTypeT = SourceNodeTypeT>
    void make_dynamic_edge(std::shared_ptr<segment::ObjectProperties> source,
                           std::shared_ptr<segment::ObjectProperties> sink)
    {
        node::make_edge(source->source_typed<SourceNodeTypeT>(), sink->sink_typed<SinkNodeTypeT>());
    }

    template <typename ObjectT>
    void add_throughput_counter(std::shared_ptr<segment::Object<ObjectT>> segment_object)
    {
        auto runnable = std::dynamic_pointer_cast<Runnable<ObjectT>>(segment_object);
        CHECK(runnable);
        CHECK(segment_object->is_source());
        using source_type_t = typename ObjectT::source_type_t;
        auto counter        = m_backend.make_throughput_counter(runnable->name());
        runnable->object().add_epilogue_tap([counter](const source_type_t& data) { counter(1); });
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
        runnable->object().add_epilogue_tap([counter, tick_fn](const source_type_t& data) { counter(tick_fn(data)); });
    }

  private:
    using sp_segment_module_t = std::shared_ptr<mrc::modules::SegmentModule>;
    using sp_obj_prop_t       = std::shared_ptr<segment::ObjectProperties>;

    std::string m_namespace_prefix;
    std::vector<std::string> m_namespace_stack{};
    std::vector<sp_segment_module_t> m_module_stack{};

    internal::segment::IBuilder& m_backend;

    void ns_push(sp_segment_module_t module);
    void ns_pop();

    friend Definition;
};

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

        m_backend.add_runnable(name, segment_node);
        m_backend.add_object(name, segment_node);
        segment_object = segment_node;
    }
    else
    {
        auto segment_node = std::make_shared<Component<ObjectT>>(std::move(node));
        m_backend.add_object(name, segment_node);
        segment_object = segment_node;
    }

    CHECK(segment_object);
    return segment_object;
}

template <typename T>
std::shared_ptr<Object<node::SinkProperties<T>>> Builder::get_egress(std::string name)
{
    auto base = m_backend.get_egress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Egress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<Object<node::SinkProperties<T>>>(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

template <typename T>
std::shared_ptr<Object<node::SourceProperties<T>>> Builder::get_ingress(std::string name)
{
    auto base = m_backend.get_ingress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Ingress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<Object<node::SourceProperties<T>>>(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Ingress port type mismatch: " + name);
    }

    return port;
}

}  // namespace mrc::segment
