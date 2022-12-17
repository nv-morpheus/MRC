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

#include "pymrc/segment.hpp"

#include "pymrc/node.hpp"
#include "pymrc/operators.hpp"
#include "pymrc/subscriber.hpp"
#include "pymrc/types.hpp"
#include "pymrc/utilities/acquire_gil.hpp"
#include "pymrc/utilities/function_wrappers.hpp"
#include "pymrc/utils.hpp"

#include "mrc/node/edge_builder.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/segment/builder.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

// IWYU thinks we need array for py::print
// IWYU pragma: no_include <array>

namespace mrc::pymrc {

namespace py = pybind11;

std::shared_ptr<mrc::segment::ObjectProperties> build_source(mrc::segment::Builder& self,
                                                             const std::string& name,
                                                             std::function<py::iterator()> iter_factory)
{
    auto wrapper = [iter_factory](PyObjectSubscriber& subscriber) mutable {
        auto& ctx = runnable::Context::get_runtime_context();

        AcquireGIL gil;

        try
        {
            DVLOG(10) << ctx.info() << " Starting source";

            // Get the iterator from the factory
            auto iter = iter_factory();

            // Loop over the iterator
            while (iter != py::iterator::sentinel())
            {
                // Get the next value
                auto next_val = py::cast<py::object>(*iter);

                {
                    // Release the GIL to call on_next
                    pybind11::gil_scoped_release nogil;

                    //  Only send if its subscribed. Very important to ensure the object has been moved!
                    if (subscriber.is_subscribed())
                    {
                        subscriber.on_next(std::move(next_val));
                    }
                }

                // Increment it for next loop
                ++iter;
            }

        } catch (const std::exception& e)
        {
            LOG(ERROR) << ctx.info() << "Error occurred in source. Error msg: " << e.what();

            gil.release();
            subscriber.on_error(std::current_exception());
            return;
        }

        // Release the GIL to call on_complete
        gil.release();

        subscriber.on_completed();

        DVLOG(10) << ctx.info() << " Source complete";
    };

    return self.construct_object<PythonSource<PyHolder>>(name, wrapper);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::iterator source_iterator)
{
    // Capture the generator factory
    return build_source(self, name, [iterator = PyObjectHolder(std::move(source_iterator))]() mutable {
        // Check if the iterator has been started already
        if (!iterator)
        {
            LOG(ERROR)
                << "Cannot have multiple progress engines for the iterator overload. Iterators cannot be duplicated";
            throw std::runtime_error(
                "Cannot have multiple progress engines for the iterator overload. Iterators cannot be duplicated");
        }

        // Move the object into the iterator to ensure its only used once.
        return py::cast<py::iterator>(py::object(std::move(iterator)));
    });
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::iterable source_iterable)
{
    // Capture the iterator
    return build_source(self, name, [iterable = PyObjectHolder(std::move(source_iterable))]() {
        // Turn the iterable into an iterator
        return py::iter(iterable);
    });
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::function gen_factory)
{
    // Capture the generator factory
    return build_source(self, name, [gen_factory = PyObjectHolder(std::move(gen_factory))]() {
        // Call the generator factory to make a new generator
        return py::cast<py::iterator>(gen_factory());
    });
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_sink(mrc::segment::Builder& self,
                                                                        const std::string& name,
                                                                        OnNextFunction on_next,
                                                                        OnErrorFunction on_error,
                                                                        OnCompleteFunction on_completed)
{
    return self.make_sink<PyHolder, PythonSink>(name, on_next, on_error, on_completed);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_sink_component(mrc::segment::Builder& self,
                                                                                  const std::string& name,
                                                                                  OnNextFunction on_next,
                                                                                  OnErrorFunction on_error,
                                                                                  OnCompleteFunction on_completed)
{
    return self.make_sink_component<PyHolder, PythonSinkComponent>(name, on_next, on_error, on_completed);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::get_ingress(mrc::segment::Builder& self,
                                                                          const std::string& name)
{
    auto it_caster = node::PortRegistry::s_port_to_type_index.find(name);
    if (it_caster != node::PortRegistry::s_port_to_type_index.end())
    {
        VLOG(2) << "Found an ingress port caster for " << name;

        return self.get_ingress(name, it_caster->second);
    }
    return self.get_ingress<PyHolder>(name);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::get_egress(mrc::segment::Builder& self,
                                                                         const std::string& name)
{
    auto it_caster = node::PortRegistry::s_port_to_type_index.find(name);
    if (it_caster != node::PortRegistry::s_port_to_type_index.end())
    {
        VLOG(2) << "Found an egress port caster for " << name;

        return self.get_egress(name, it_caster->second);
    }

    return self.get_egress<PyHolder>(name);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node(mrc::segment::Builder& self,
                                                                        const std::string& name,
                                                                        OnDataFunction on_data)
{
    show_deprecation_warning(
        "Passing a map function object to make_node() is deprecated and will be removed in a future version. "
        "make_node() now requires an operator. Use "
        "make_node(name, mrc.core.operators.map(map_fn)) instead.");

    return BuilderProxy::make_node(self, name, py::args(py::make_tuple(py::cast(OperatorsProxy::map(on_data)))));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node(mrc::segment::Builder& self,
                                                                        const std::string& name,
                                                                        pybind11::args operators)
{
    auto node = self.make_node<PyHolder, PyHolder, PythonNode>(name);

    node->object().make_stream(
        [operators = PyObjectHolder(std::move(operators))](const PyObjectObservable& input) -> PyObjectObservable {
            AcquireGIL gil;

            // Call the pipe function to convert all of the args to a new observable
            return ObservableProxy::pipe(&input, py::cast<py::args>(operators));
        });

    return node;
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node_full(
    mrc::segment::Builder& self,
    const std::string& name,
    std::function<void(const pymrc::PyObjectObservable& obs, pymrc::PyObjectSubscriber& sub)> sub_fn)
{
    show_deprecation_warning(
        "make_node_full(name, sub_fn) is deprecated and will be removed in a future version. Use "
        "make_node(name, mrc.core.operators.build(sub_fn)) instead.");

    auto node = self.make_node<PyHolder, PyHolder, PythonNode>(name);

    node->object().make_stream([sub_fn](const PyObjectObservable& input) -> PyObjectObservable {
        return rxcpp::observable<>::create<PyHolder>([input, sub_fn](pymrc::PyObjectSubscriber output) {
            try
            {
                py::gil_scoped_acquire gil;

                // Call the subscribe function
                sub_fn(input, output);

                return output;

            } catch (py::error_already_set& err)
            {
                LOG(ERROR) << "Python occurred during full node subscription. Error: " + std::string(err.what());

                // Rethrow python exceptions
                throw;
            } catch (std::exception& err)
            {
                LOG(ERROR) << "Exception occurred during subscription. Error: " + std::string(err.what());
                throw;
            }
        });
    });

    return node;
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node_component(mrc::segment::Builder& self,
                                                                                  const std::string& name,
                                                                                  pybind11::args operators)
{
    auto node = self.make_node_component<PyHolder, PyHolder, PythonNodeComponent>(name);

    node->object().make_stream(
        [operators = PyObjectHolder(std::move(operators))](const PyObjectObservable& input) -> PyObjectObservable {
            // Call the pipe function to convert all of the args to a new observable
            return ObservableProxy::pipe(&input, py::cast<py::args>(operators));
        });

    return node;
}

std::shared_ptr<mrc::modules::SegmentModule> BuilderProxy::load_module_from_registry(
    mrc::segment::Builder& self,
    const std::string& module_id,
    const std::string& registry_namespace,
    std::string module_name,
    py::dict config)
{
    auto json_config = cast_from_pyobject(config);

    return self.load_module_from_registry(
        module_id, registry_namespace, std::move(module_name), std::move(json_config));
}

void BuilderProxy::init_module(mrc::segment::Builder& self, std::shared_ptr<mrc::modules::SegmentModule> module)
{
    self.init_module(module);
}

void BuilderProxy::register_module_input(mrc::segment::Builder& self,
                                         std::string input_name,
                                         std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_input(std::move(input_name), object);
}

void BuilderProxy::register_module_output(mrc::segment::Builder& self,
                                          std::string output_name,
                                          std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_output(std::move(output_name), object);
}

py::dict BuilderProxy::get_current_module_config(mrc::segment::Builder& self)
{
    auto json_config = self.get_current_module_config();

    return cast_from_json(json_config);
}

void BuilderProxy::make_edge(mrc::segment::Builder& self,
                             std::shared_ptr<mrc::segment::ObjectProperties> source,
                             std::shared_ptr<mrc::segment::ObjectProperties> sink)
{
    if (source->is_ingress_acceptor() && sink->is_ingress_provider())
    {
        node::make_edge_typeless(source->ingress_acceptor_base(), sink->ingress_provider_base());
    }
    else if (source->is_ingress_acceptor() && sink->is_ingress_provider())
    {
        node::make_edge_typeless(source->egress_provider_base(), sink->egress_acceptor_base());
    }
    else
    {
        throw std::runtime_error(
            "Invalid edges. Arguments to make_edge were incorrect. Ensure you are providing either "
            "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

}  // namespace mrc::pymrc
