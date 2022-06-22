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

#include <pysrf/segment.hpp>

#include <pysrf/edge_adaptor.hpp>
#include <pysrf/node.hpp>
#include <pysrf/types.hpp>
#include <pysrf/utils.hpp>
#include <srf/channel/status.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>
#include <srf/runnable/context.hpp>
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-operators.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx.hpp>  // IWYU pragma: keep

#include <exception>
#include <fstream>  // IWYU pragma: keep
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU thinks we need array for py::print
// IWYU pragma: no_include <array>

namespace srf::pysrf {

namespace py = pybind11;

std::shared_ptr<srf::segment::ObjectProperties> build_source(srf::segment::Builder& self,
                                                             const std::string& name,
                                                             std::function<py::iterator()> iter_factory)
{
    auto wrapper = [iter_factory](PyObjectSubscriber& s) mutable {
        auto& ctx = runnable::Context::get_runtime_context();

        AcquireGIL gil;

        try
        {
            DVLOG(10) << ctx.info() << " Starting source";

            // Get the iterator from the factory
            auto it = iter_factory();

            // Loop over the iterator
            while (it != py::iterator::sentinel())
            {
                // Get the next value
                auto next_val = py::cast<py::object>(*it);

                // Increment it for next loop
                ++it;

                {
                    // Release the GIL to call on_next
                    pybind11::gil_scoped_release nogil;

                    //  Only send if its subscribed. Very important to ensure the object has been moved!
                    if (s.is_subscribed())
                    {
                        s.on_next(std::move(next_val));
                    }
                }
            }

        } catch (const std::exception& e)
        {
            LOG(ERROR) << ctx.info() << "Error occurred in source. Error msg: " << e.what();

            gil.release();
            s.on_error(std::current_exception());
            return;
        }

        // Release the GIL to call on_complete
        gil.release();

        s.on_completed();

        DVLOG(10) << ctx.info() << " Source complete";
    };

    auto node = self.construct_object<PythonSource<PyHolder>>(name, wrapper);

    return node;
}

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_source(srf::segment::Builder& self,
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

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_source(srf::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::iterable source_iterable)
{
    // Capture the iterator
    return build_source(self, name, [iterable = PyObjectHolder(std::move(source_iterable))]() {
        // Turn the iterable into an iterator
        return py::iter(iterable);
    });
}

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_source(srf::segment::Builder& self,
                                                                          const std::string& name,
                                                                          py::function gen_factory)
{
    // Capture the generator factory
    return build_source(self, name, [gen_factory = PyObjectHolder(std::move(gen_factory))]() {
        // Call the generator factory to make a new generator
        return py::cast<py::iterator>(gen_factory());
    });
}

// std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_source(
//     srf::segment::Builder& self, const std::string& name, const std::function<void(pysrf::PyObjectSubscriber& sub)>&
//     f)
// {
//     auto wrapper = [f](pysrf::PyObjectSubscriber& s) {
//         py::gil_scoped_acquire gil;

//         try
//         {
//             f(s);
//         } catch (py::error_already_set& err)
//         {
//             py::print("Error hit!");
//             py::print(err.what());
//             throw;  // Rethrow to propagate back to python
//         }
//     };

//     return self.construct_object<PythonSource<py::object>>(name, wrapper);
// }

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_sink(srf::segment::Builder& self,
                                                                        const std::string& name,
                                                                        std::function<void(py::object x)> on_next,
                                                                        std::function<void(py::object x)> on_error,
                                                                        std::function<void()> on_completed)
{
    auto on_next_w = [on_next](PyHolder x) {
        pybind11::gil_scoped_acquire gil;
        on_next(std::move(x));  // Move the object into a temporary
    };

    auto on_error_w = [on_error](std::exception_ptr x) {
        pybind11::gil_scoped_acquire gil;

        // First, translate the exception setting the python exception value
        py::detail::translate_exception(x);

        // Creating py::error_already_set will clear the exception and retrieve the value
        py::error_already_set active_ex;

        // Now actually pass the exception to the callback
        on_error(active_ex.value());
    };

    auto on_completed_w = [on_completed]() {
        pybind11::gil_scoped_acquire gil;
        on_completed();
    };

    auto node = self.construct_object<PythonSink<PyHolder>>(
        name, rxcpp::make_observer<PyHolder>(on_next_w, on_error_w, on_completed_w));

    return node;
}

/*
std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::construct_object(srf::segment::Builder& self,
                                                                        const std::string& name,
                                                                        std::function<py::object(py::object x)> map_f)
{
    return self.construct_object<PythonNode<py::object, py::object>>(
        name, [map_f](srf::Observable<py::object>& input, pysrf::PyObjectSubscriber& output) {
            return input.subscribe(srf::make_observer<py::object>(
                [map_f, &output](py::object&& x) {
                    // Since the argument cant be py::object&&, steal the reference here to
                    // prevent incrementing the ref count without the GIL auto stolen =
                    // py::reinterpret_steal<py::object>(x);
                    py::object returned;
                    try
                    {
                        // Acquire the GIL here
                        py::gil_scoped_acquire gil;

                        // Call the map function
                        returned = map_f(std::move(x));

                        // While we have the GIL, check for downstream subscriptions.
                        if (!output.is_subscribed())
                        {
                            // This object needs to lose its ref count while we have the GIL
                            py::object tmp = std::move(returned);
                        }

                        // Release the GIL before calling on_next to prevent deadlocks
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

                    if (returned)
                    {
                        // Make sure to move here since we dont have the GIL
                        output.on_next(std::move(returned));
                        assert(!returned);
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        });
}
*/

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_node(
    srf::segment::Builder& self, const std::string& name, std::function<pybind11::object(pybind11::object x)> map_f)
{
    auto node = self.construct_object<PythonNode<PyHolder, PyHolder>>(
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

    return node;
}

std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_node_full(
    srf::segment::Builder& self,
    const std::string& name,
    std::function<void(const pysrf::PyObjectObservable& obs, pysrf::PyObjectSubscriber& sub)> sub_fn)
{
    auto node = self.construct_object<PythonNode<PyHolder, PyHolder>>(name);

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

// void SegmentProxy::test_fn(srf::segment::Builder& self, py::function py_func)
// {
//     auto inspect = py::module_::import("inspect");

//     auto fn_sig = inspect.attr("signature")(py_func);

//     auto return_annotation = fn_sig.attr("return_annotation")();

//     // Debug print
//     py::print("in test_fn");
// }

void SegmentProxy::make_py2cxx_edge_adapter(srf::segment::Builder& self,
                                            std::shared_ptr<srf::segment::ObjectProperties> source,
                                            std::shared_ptr<srf::segment::ObjectProperties> sink,
                                            py::object& sink_t)
{
    using source_type_t = py::object;

    /*

    // https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
    pybind11::dtype dtype = pybind11::dtype::from_args(sink_t);
    switch (dtype.kind())
    {
    case 'b':
        self.make_dynamic_edge<source_type_t, bool, false>(source->name(), sink->name());
        break;
    case 'i':
        if (dtype.itemsize() == 4)
        {
            self.make_dynamic_edge<source_type_t, int32_t, false>(source->name(), sink->name());
            break;
        }
        self.make_dynamic_edge<source_type_t, int64_t, false>(source->name(), sink->name());
        break;
    case 'u':
        if (dtype.itemsize() == 4)
        {
            self.make_dynamic_edge<source_type_t, uint32_t, false>(source->name(), sink->name());
            break;
        }
        self.make_dynamic_edge<source_type_t, uint64_t, false>(source->name(), sink->name());
        break;
    case 'f':
        if (dtype.itemsize() == 4)
        {
            self.make_dynamic_edge<source_type_t, float, false>(source->name(), sink->name());
            break;
        }
        self.make_dynamic_edge<source_type_t, double, false>(source->name(), sink->name());
        break;
    case 'c':
        throw std::runtime_error("Complex-float datatypes are not currently supported");
    case 'm':
        throw std::runtime_error("Timedelta datatypes are not currently supported");
    case 'M':
        throw std::runtime_error("Datetime datatypes are not currently supported");
    case 'O':
        throw std::runtime_error("Automatic conversion between py::objects is not supported.");
    case 'S':
        self.make_dynamic_edge<source_type_t, std::string, false>(source->name(), sink->name());
        break;
    case 'U':
        self.make_dynamic_edge<source_type_t, std::string, false>(source->name(), sink->name());
        break;
    case 'V':
        throw std::runtime_error("Void datatypes are not supported");
    default:
        throw std::runtime_error("Unknown sink type");
    }

    */
}

void SegmentProxy::make_cxx2py_edge_adapter(srf::segment::Builder& self,
                                            std::shared_ptr<srf::segment::ObjectProperties> source,
                                            std::shared_ptr<srf::segment::ObjectProperties> sink,
                                            py::object& source_t)
{
    using sink_type_t = pybind11::object;

    LOG(FATAL) << "fixme";
    /*
        // https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        pybind11::dtype dtype = pybind11::dtype::from_args(source_t);
        switch (dtype.kind())
        {
        case 'b':
            self.make_dynamic_edge<bool, sink_type_t, false>(source->name(), sink->name());
            break;
        case 'i':
            if (dtype.itemsize() == 4)
            {
                self.make_dynamic_edge<int32_t, sink_type_t, false>(source->name(), sink->name());
                break;
            }
            self.make_dynamic_edge<int64_t, sink_type_t, false>(source->name(), sink->name());
            break;
        case 'u':
            if (dtype.itemsize() == 4)
            {
                self.make_dynamic_edge<uint32_t, sink_type_t, false>(source->name(), sink->name());
                break;
            }
            self.make_dynamic_edge<uint64_t, sink_type_t, false>(source->name(), sink->name());
            break;
        case 'f':
            if (dtype.itemsize() == 4)
            {
                self.make_dynamic_edge<float, sink_type_t, false>(source->name(), sink->name());
                break;
            }
            self.make_dynamic_edge<double, sink_type_t, false>(source->name(), sink->name());
            break;

        case 'c':
            throw std::runtime_error("Complex-float datatypes are not supported.");
        case 'm':
            throw std::runtime_error("Timedelta datatypes are not supported.");
        case 'M':
            throw std::runtime_error("Datetime datatypes are not supported.");
        case 'O':
            throw std::runtime_error("Automatic conversion to generic py::object is not supported.");
        case 'S':
            self.make_dynamic_edge<std::string, sink_type_t, false>(source->name(), sink->name());
        case 'U':
            self.make_dynamic_edge<std::string, sink_type_t, false>(source->name(), sink->name());
            break;
        case 'V':
            throw std::runtime_error("Void datatypes are not supported");
        default:
            throw std::runtime_error("Unknown sink type");
        }
    */
}

void SegmentProxy::make_edge(srf::segment::Builder& self,
                             std::shared_ptr<srf::segment::ObjectProperties> source,
                             std::shared_ptr<srf::segment::ObjectProperties> sink)
{
    node::EdgeBuilder::make_edge_typeless(source->source_base(), sink->sink_base());
}

// std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::make_file_reader(srf::segment::Builder& self,
//                                                                                const std::string& name,
//                                                                                const std::string& filename)
// {
//     return self.construct_object<PythonSource<std::string>>(name, [filename](rxcpp::subscriber<std::string>& s) {
//         std::ifstream file(filename);
//         std::string line;

//         // While we are running and there are still lines to read in
//         // the file
//         try
//         {
//             while (s.is_subscribed() && std::getline(file, line))
//             {
//                 // Push to downstream
//                 s.on_next(line);
//             }
//         } catch (...)
//         {
//             s.on_error(std::current_exception());
//         }

//         DVLOG(5) << "Input file complete" << std::endl;
//         s.on_completed();
//     });
// }

// std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::debug_float_source(srf::segment::Builder& self,
//                                                                                  const std::string& name,
//                                                                                  std::size_t iterations)
// {
//     return self.make_source<double>(name, [iterations](rxcpp::subscriber<double> sub) {
//         auto i = 0;
//         while (sub.is_subscribed() && i < iterations)
//         {
//             sub.on_next(std::atan(1) * 4);
//             i++;
//         }

//         sub.on_completed();
//     });
// }

/*
std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::debug_float_passthrough(srf::segment::Builder& self,
                                                                                      const std::string& name)
{
    return self.construct_object<PythonNode, double, double>(name, rxcpp::operators::map<double>([](double d) { return
d; }));
}
*/

// std::shared_ptr<PyNode> SegmentProxy::flatten_list(srf::segment::Builder& self, const std::string& name)
// {
//     auto flatten_node = self.make_node<py::object, std::string>(
//         name,
//         rxcpp::operators::concat_map([](py::object wrapper_thing) {
//             py::gil_scoped_acquire gil;
//             auto info = py::str("Concat map got object: ") + py::str(wrapper_thing);
//             py::print(info);
//             info = py::str("concat map Object ref count: ") + py::str(std::to_string(wrapper_thing.ref_count()));
//             py::print(info);

//             return rxcpp::observable<>::create<std::string>([wrapper_thing](rxcpp::subscriber<std::string> sub) {
//                 pybind11::gil_scoped_acquire gil;
//                 py::print("Entering concat map");
//                 try
//                 {
//                     std::cerr << "Moving vector value" << std::endl;
//                     // sub.on_next(std::move(x.m_vector_str.back()));
//                     auto s = pybind11::cast<std::string>(wrapper_thing);
//                     {
//                         pybind11::gil_scoped_release rel;
//                         sub.on_next(std::move(s));
//                     }
//                 } catch (...)
//                 {
//                     std::cerr << "Caught exception" << std::endl;
//                     sub.on_error(rxcpp::util::current_exception());
//                 }

//                 {
//                     // Release the GIL to call on_complete
//                     std::cerr << "Calling concat_map on complete" << std::endl;
//                     sub.on_completed();
//                     std::cerr << "On_completed call finished." << std::endl;
//                 }
//             });
//         }),
//         rxcpp::operators::map([](const std::string& s) { return s; }));

//     return std::static_pointer_cast<PyNode>(flatten_node);
// }

/*
std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::debug_string_passthrough(srf::segment::Builder& self,
                                                                                       const std::string& name)
{
    return self.construct_object<PythonNode, std::string, std::string>(
        name, srf::operators::map<std::string, std::string>([](std::string s) { return s; }));
}
*/

// std::shared_ptr<srf::segment::ObjectProperties> SegmentProxy::debug_float_sink(srf::segment::Builder& self,
//                                                                                const std::string& name)
// {
//     return self.construct_object<PythonSink<double>>(std::move(name), [](double d) {});
// }
}  // namespace srf::pysrf
