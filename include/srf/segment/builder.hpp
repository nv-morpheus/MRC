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

#include <srf/engine/segment/ibuilder.hpp>
#include <srf/exceptions/runtime_error.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/rx_node.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/rx_source.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>
#include <srf/runnable/launchable.hpp>
#include <srf/runnable/runnable.hpp>
#include <srf/segment/component.hpp>
#include <srf/segment/egress_port.hpp>
#include <srf/segment/forward.hpp>
#include <srf/segment/object.hpp>
#include <srf/segment/runnable.hpp>
#include <srf/utils/macros.hpp>

#include <glog/logging.h>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-observer.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace srf::segment {

class Builder final : private internal::segment::IBuilder
{
    Builder(internal::segment::IBuilder& backend);

  public:
    ~Builder() final = default;

    DELETE_COPYABILITY(Builder);
    DELETE_MOVEABILITY(Builder);

    template <typename T>
    std::shared_ptr<Object<node::SinkProperties<T>>> get_egress(std::string name);

    template <typename T>
    std::shared_ptr<Object<node::SourceProperties<T>>> get_ingress(std::string name);

    template <typename ObjectT>
    std::shared_ptr<Object<ObjectT>> make_object(std::string name, std::unique_ptr<ObjectT> node);

    template <typename ObjectT, typename... ArgsT>
    std::shared_ptr<Object<ObjectT>> construct_object(std::string name, ArgsT&&... args)
    {
        return make_object(std::move(name), std::make_unique<ObjectT>(std::forward<ArgsT>(args)...));
    }

    template <typename SourceTypeT, typename CreateFnT>
    auto make_source(std::string name, CreateFnT&& create_fn)
    {
        return make_object(std::move(name),
                           std::make_unique<node::RxSource<SourceTypeT>>(
                               rxcpp::observable<>::create<SourceTypeT>(std::forward<CreateFnT>(create_fn))));
    }

    template <typename SinkTypeT, typename... ArgsT>
    auto make_sink(std::string name, ArgsT&&... ops)
    {
        return make_object(std::move(name),
                           std::make_unique<node::RxSink<SinkTypeT>>(
                               rxcpp::make_observer_dynamic<SinkTypeT>(std::forward<ArgsT>(ops)...)));
    }

    template <typename SinkTypeT, typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops)
    {
        return make_object(std::move(name),
                           std::make_unique<node::RxNode<SinkTypeT, SinkTypeT>>(std::forward<ArgsT>(ops)...));
    }

    template <typename SinkTypeT, typename SourceTypeT, typename... ArgsT>
    auto make_node(std::string name, ArgsT&&... ops)
    {
        return make_object(std::move(name),
                           std::make_unique<node::RxNode<SinkTypeT, SourceTypeT>>(std::forward<ArgsT>(ops)...));
    }

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

    template <typename SourceNodeTypeT, typename SinkNodeTypeT = SourceNodeTypeT>
    void make_dynamic_edge(const std::string& source_name, const std::string& sink_name)
    {
        auto& source_obj = find_object(source_name);
        auto& sink_obj   = find_object(sink_name);
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
        auto counter        = make_throughput_counter(runnable->name());
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
        auto counter        = make_throughput_counter(runnable->name());
        runnable->object().add_epilogue_tap([counter, tick_fn](const source_type_t& data) { counter(tick_fn(data)); });
    }

  private:
    const std::string& name() const final;
    bool has_object(const std::string& name) const final;
    ::srf::segment::ObjectProperties& find_object(const std::string& name) final;
    void add_object(const std::string& name, std::shared_ptr<::srf::segment::ObjectProperties> object) final;
    void add_runnable(const std::string& name, std::shared_ptr<runnable::Launchable> runnable) final;
    std::shared_ptr<::srf::segment::IngressPortBase> get_ingress_base(const std::string& name) final;
    std::shared_ptr<::srf::segment::EgressPortBase> get_egress_base(const std::string& name) final;

    std::function<void(std::int64_t)> make_throughput_counter(const std::string& name) final;

    internal::segment::IBuilder& m_backend;

    friend Definition;
};

template <typename ObjectT>
std::shared_ptr<Object<ObjectT>> Builder::make_object(std::string name, std::unique_ptr<ObjectT> node)
{
    if (has_object(name))
    {
        LOG(ERROR) << "A Object named " << name << " is already registered";
        throw exceptions::SrfRuntimeError("duplicate name detected - name owned by a node");
    }

    std::shared_ptr<Object<ObjectT>> segment_object{nullptr};

    if constexpr (std::is_base_of_v<runnable::Runnable, ObjectT>)
    {
        auto segment_name = this->name() + "/" + name;
        auto segment_node = std::make_shared<Runnable<ObjectT>>(segment_name, std::move(node));
        add_runnable(name, segment_node);
        add_object(name, segment_node);
        segment_object = segment_node;
    }
    else
    {
        auto segment_node = std::make_shared<Component<ObjectT>>(std::move(node));
        add_object(name, segment_node);
        segment_object = segment_node;
    }

    CHECK(segment_object);
    return segment_object;
}

template <typename T>
std::shared_ptr<Object<node::SinkProperties<T>>> Builder::get_egress(std::string name)
{
    auto base = get_egress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("egress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<EgressPort<T>>(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("egress port type mismatch: " + name);
    }

    return port;
}

template <typename T>
std::shared_ptr<Object<node::SourceProperties<T>>> Builder::get_ingress(std::string name)
{
    auto base = get_ingress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("ingress port name not found: " + name);
    }

    auto port = std::dynamic_pointer_cast<IngressPort<T>>(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("ingress port type mismatch: " + name);
    }

    return port;
}

}  // namespace srf::segment
