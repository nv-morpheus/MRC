/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/types.hpp"

#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/segment/builder.hpp"  // IWYU pragma: keep

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace mrc::modules {
class SegmentModule;
}  // namespace mrc::modules

namespace mrc::pymrc {
struct OnCompleteFunction;
}
namespace mrc::pymrc {
struct OnDataFunction;
}
namespace mrc::pymrc {
struct OnErrorFunction;
}
namespace mrc::pymrc {
struct OnNextFunction;
}
namespace mrc::segment {
struct ObjectProperties;
}

namespace mrc::pymrc {
// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

/**
 * Relates to https://github.com/pybind/pybind11/issues/1241 -- for a general solution see pydrake's WrapFunction
 *  method.
 *
 * We need to force pybind to pass us a function that expects a mrc::segment::IBuilder* not a mrc::segment::IBuilder&.
 * If not it'll try to make a copy and mrc::segment::IBuilder isnt' copy-constructable. Once we have that, we wrap it
 * with our reference based function.
 *
 * @tparam ClassT Class where the init method binding is defined.
 * @tparam ArgsT any additional arguments to pass to the initializer function
 * @param method method of ClassT that we need to wrap.
 * @return wrapped ClassT::*method function.
 */
template <typename ClassT, typename... ArgsT>
auto wrap_segment_init_callback(void (ClassT::*method)(const std::string&,
                                                       const std::function<void(mrc::segment::IBuilder&, ArgsT...)>&))
{
    // Build up the function we're going to return, the signature on this function is what forces python to give us
    //  a pointer.
    auto func = [method](ClassT* self,
                         const std::string& name,
                         const std::function<void(mrc::segment::IBuilder*, ArgsT...)>& f_to_wrap) {
        auto f_wrapped = [f_to_wrap](mrc::segment::IBuilder& t, ArgsT... args) {
            f_to_wrap(&t, std::forward<ArgsT>(args)...);
        };

        return (self->*method)(std::forward<const std::string&>(name), (std::forward<decltype(f_wrapped)>(f_wrapped)));
    };

    return func;
}

/**
 * [Overload for segment initialization with additional port name + type information store in pybind11::lists.]
 *
 * Relates to https://github.com/pybind/pybind11/issues/1241 -- for a general solution see pydrake's WrapFunction
 *  method.
 *
 * We need to force pybind to pass us a function that expects a mrc::segment::IBuilder* not a mrc::segment::IBuilder&.
 * If not it'll try to make a copy and mrc::segment::IBuilder isn't copy-constructable. Once we have that, we wrap it
 * with our reference based function.
 *
 * @tparam ClassT Class where the init method binding is defined.
 * @tparam ArgsT any additional arguments to pass to the initializer function
 * @param method method of ClassT that we need to wrap.
 * @return wrapped ClassT::*method function.
 */
template <typename ClassT, typename... ArgsT>
auto wrap_segment_init_callback(
    void (ClassT::*method)(const std::string&,
                           pybind11::list,
                           pybind11::list,
                           const std::function<void(mrc::segment::IBuilder&, ArgsT... args)>&))
{
    // Build up the function we're going to return, the signature on this function is what forces python to give us
    //  a pointer.
    auto func = [method](ClassT* self,
                         const std::string& name,
                         pybind11::list ingress_port_ids,
                         pybind11::list egress_port_ids,
                         const std::function<void(mrc::segment::IBuilder*, ArgsT...)>& f_to_wrap) {
        auto f_wrapped = [f_to_wrap](mrc::segment::IBuilder& t, ArgsT... args) {
            f_to_wrap(&t, std::forward<ArgsT>(args)...);
        };

        return (self->*method)(std::forward<const std::string&>(name),
                               std::forward<pybind11::list>(ingress_port_ids),
                               std::forward<pybind11::list>(egress_port_ids),
                               std::forward<decltype(f_wrapped)>(f_wrapped));
    };

    return func;
}

class BuilderProxy
{
  public:
    static std::shared_ptr<mrc::segment::ObjectProperties> make_source(mrc::segment::IBuilder& self,
                                                                       const std::string& name,
                                                                       pybind11::iterator source_iterator);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source(mrc::segment::IBuilder& self,
                                                                       const std::string& name,
                                                                       pybind11::iterable source_iter);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source(mrc::segment::IBuilder& self,
                                                                       const std::string& name,
                                                                       pybind11::function gen_factory);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source(
        mrc::segment::IBuilder& self,
        const std::string& name,
        const std::function<void(pymrc::PyObjectSubscriber& sub)>& f);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source_component(mrc::segment::IBuilder& self,
                                                                                 const std::string& name,
                                                                                 pybind11::iterator source_iterator);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source_component(mrc::segment::IBuilder& self,
                                                                                 const std::string& name,
                                                                                 pybind11::iterable source_iter);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_source_component(mrc::segment::IBuilder& self,
                                                                                 const std::string& name,
                                                                                 pybind11::function gen_factory);

    /**
     * Construct a new pybind11::object sink.
     * Create and return a Segment node used to sink python objects following out of the Segment.
     *
     * (py) @param name: Unique name of the node that will be created in the MRC Segment.
     * (py) @param on_next: python/std function that will be called on a new data element.
     * (py) @param on_error: python/std function that will be called if an error occurs.
     * (py) @param on_completed: python/std function that will be called
     *  Python example.
     *  ```python
     *      def my_on_next(x):
     *          print(f"Sinking {x}")
     *      def my_on_error(err):
     *          print(f"An error occurred: {err}")
     *      def my_on_completed():
     *          print(f"Completed processing")
     *
     *      sink = segment.make_sink("test", my_on_next, my_on_error, my_on_completed)
     *  ```
     */
    static std::shared_ptr<mrc::segment::ObjectProperties> make_sink(mrc::segment::IBuilder& self,
                                                                     const std::string& name,
                                                                     OnNextFunction on_next,
                                                                     OnErrorFunction on_error,
                                                                     OnCompleteFunction on_completed);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_sink_component(mrc::segment::IBuilder& self,
                                                                               const std::string& name,
                                                                               OnNextFunction on_next,
                                                                               OnErrorFunction on_error,
                                                                               OnCompleteFunction on_completed);

    // Deprecated. This must come first
    static std::shared_ptr<mrc::segment::ObjectProperties> make_node(mrc::segment::IBuilder& self,
                                                                     const std::string& name,
                                                                     OnDataFunction on_data);

    /**
     * Construct a new 'pure' python::object -> python::object node
     *
     * This will create and return a new lambda function with the following signature:
     * (py) @param name : Unique name of the node that will be created in the MRC Segment.
     */
    static std::shared_ptr<mrc::segment::ObjectProperties> make_node(mrc::segment::IBuilder& self,
                                                                     const std::string& name,
                                                                     pybind11::args operators);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_node_full(
        mrc::segment::IBuilder& self,
        const std::string& name,
        std::function<void(const pymrc::PyObjectObservable& obs, pymrc::PyObjectSubscriber& sub)> sub_fn);

    static std::shared_ptr<mrc::segment::ObjectProperties> make_node_component(mrc::segment::IBuilder& self,
                                                                               const std::string& name,
                                                                               pybind11::args operators);

    static void make_edge(mrc::segment::IBuilder& self,
                          std::variant<std::shared_ptr<mrc::segment::ObjectProperties>,
                                       std::shared_ptr<mrc::edge::IWritableAcceptorBase>,
                                       std::shared_ptr<mrc::edge::IReadableProviderBase>> source,
                          std::variant<std::shared_ptr<mrc::segment::ObjectProperties>,
                                       std::shared_ptr<mrc::edge::IWritableProviderBase>,
                                       std::shared_ptr<mrc::edge::IReadableAcceptorBase>> sink);

    static void splice_edge(mrc::segment::IBuilder& self,
                            std::string& source,
                            std::string& sink,
                            std::shared_ptr<mrc::segment::ObjectProperties> splice_input,
                            std::shared_ptr<mrc::segment::ObjectProperties> splice_output)
    {}

    static std::shared_ptr<mrc::segment::ObjectProperties> get_ingress(mrc::segment::IBuilder& self,
                                                                       const std::string& name);

    static std::shared_ptr<mrc::segment::ObjectProperties> get_egress(mrc::segment::IBuilder& self,
                                                                      const std::string& name);

    static std::shared_ptr<mrc::modules::SegmentModule> load_module_from_registry(mrc::segment::IBuilder& self,
                                                                                  const std::string& module_id,
                                                                                  const std::string& registry_namespace,
                                                                                  std::string module_name,
                                                                                  pybind11::dict config = {});

    static void register_module_input(mrc::segment::IBuilder& self,
                                      std::string input_name,
                                      std::shared_ptr<segment::ObjectProperties> object);

    static void register_module_output(mrc::segment::IBuilder& self,
                                       std::string output_name,
                                       std::shared_ptr<segment::ObjectProperties> object);

    static pybind11::dict get_current_module_config(mrc::segment::IBuilder& self);

    static void init_module(mrc::segment::IBuilder& self, std::shared_ptr<mrc::modules::SegmentModule> module);
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
