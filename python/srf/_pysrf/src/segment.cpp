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

#include "pysrf/segment.hpp"

#include "pysrf/node.hpp"
#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/channel/status.hpp"
#include "srf/core/utils.hpp"
#include "srf/manifold/egress.hpp"
#include "srf/modules/segment_modules.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/port_registry.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/runnable/context.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx.hpp>

#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

// IWYU thinks we need array for py::print
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <boost/fiber/future/detail/shared_state.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/task_base.hpp>
// IWYU pragma: no_include <boost/hana/if.hpp>
// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include "rx-includes.hpp"

namespace srf::pysrf {

namespace py = pybind11;

std::shared_ptr<srf::segment::ObjectProperties> build_source(srf::segment::Builder& self,
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

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_source(srf::segment::Builder& self,
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

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_source(srf::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::iterable source_iterable)
{
    // Capture the iterator
    return build_source(self, name, [iterable = PyObjectHolder(std::move(source_iterable))]() {
        // Turn the iterable into an iterator
        return py::iter(iterable);
    });
}

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_source(srf::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::function gen_factory)
{
    // Capture the generator factory
    return build_source(self, name, [gen_factory = PyObjectHolder(std::move(gen_factory))]() {
        // Call the generator factory to make a new generator
        return py::cast<py::iterator>(gen_factory());
    });
}

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_sink(srf::segment::Builder& self,
                                                                        const std::string& name,
                                                                        std::function<void(py::object object)> on_next,
                                                                        std::function<void(py::object object)> on_error,
                                                                        std::function<void()> on_completed)
{
    auto on_next_w = [on_next](PyHolder object) {
        pybind11::gil_scoped_acquire gil;
        on_next(std::move(object));  // Move the object into a temporary
    };

    auto on_error_w = [on_error](std::exception_ptr ptr) {
        pybind11::gil_scoped_acquire gil;

        // First, translate the exception setting the python exception value
        py::detail::translate_exception(ptr);

        // Creating py::error_already_set will clear the exception and retrieve the value
        py::error_already_set active_ex;

        // Now actually pass the exception to the callback
        on_error(active_ex.value());
    };

    auto on_completed_w = [on_completed]() {
        pybind11::gil_scoped_acquire gil;
        on_completed();
    };

    return self.make_sink<PyHolder, PythonSink>(name, on_next_w, on_error_w, on_completed_w);
}

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::get_ingress(srf::segment::Builder& self,
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

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::get_egress(srf::segment::Builder& self,
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

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_node(
    srf::segment::Builder& self,
    const std::string& name,
    std::function<pybind11::object(pybind11::object object)> map_f)
{
    return self.make_node<PyHolder, PyHolder, PythonNode>(
        name, rxcpp::operators::map([map_f](PyHolder data_object) -> PyHolder {
            try
            {
                py::gil_scoped_acquire gil;

                // Call the map function
                return map_f(std::move(data_object));
            } catch (py::error_already_set& err)
            {
                {
                    // Need the GIL here
                    py::gil_scoped_acquire gil;
                    py::print("Error hit!");
                    py::print(err.what());
                }

                throw;
                // caught by python output.on_error(std::current_exception());
            }
        }));
}

std::shared_ptr<srf::segment::ObjectProperties> BuilderProxy::make_node_full(
    srf::segment::Builder& self,
    const std::string& name,
    std::function<void(const pysrf::PyObjectObservable& obs, pysrf::PyObjectSubscriber& sub)> sub_fn)
{
    auto node = self.make_node<PyHolder, PyHolder, PythonNode>(name);

    node->object().make_stream([sub_fn](const PyObjectObservable& input) -> PyObjectObservable {
        return rxcpp::observable<>::create<PyHolder>([input, sub_fn](pysrf::PyObjectSubscriber output) {
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

std::shared_ptr<srf::modules::SegmentModule> BuilderProxy::load_module_from_registry(
    srf::segment::Builder& self,
    const std::string& module_id,
    const std::string& registry_namespace,
    std::string module_name,
    py::dict config)
{
    auto json_config = cast_from_pyobject(config);

    return self.load_module_from_registry(
        module_id, registry_namespace, std::move(module_name), std::move(json_config));
}

void BuilderProxy::init_module(srf::segment::Builder& self, std::shared_ptr<srf::modules::SegmentModule> module)
{
    self.init_module(module);
}

void BuilderProxy::register_module_input(srf::segment::Builder& self,
                                         std::string input_name,
                                         std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_input(std::move(input_name), object);
}

void BuilderProxy::register_module_output(srf::segment::Builder& self,
                                          std::string output_name,
                                          std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_output(std::move(output_name), object);
}

py::dict BuilderProxy::get_current_module_config(srf::segment::Builder& self)
{
    auto json_config = self.get_current_module_config();

    return cast_from_json(json_config);
}

void BuilderProxy::make_edge(srf::segment::Builder& self,
                             std::shared_ptr<srf::segment::ObjectProperties> source,
                             std::shared_ptr<srf::segment::ObjectProperties> sink)
{
    node::EdgeBuilder::make_edge_typeless(source->source_base(), sink->sink_base());
}

}  // namespace srf::pysrf
