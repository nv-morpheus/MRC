/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/operators.hpp"

#include "pymrc/types.hpp"
#include "pymrc/utils.hpp"

#include <pybind11/cast.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU pragma: no_include <array>

namespace mrc::pymrc {

namespace py = pybind11;

std::string OperatorProxy::get_name(PythonOperator& self)
{
    return self.get_name();
}

PythonOperator OperatorsProxy::filter(std::function<bool(py::object x)> filter_fn)
{
    //  Build and return the map operator
    return PythonOperator("filter", [=](PyObjectObservable source) {
        return source.filter([=](PyHolder data_object) {
            py::gil_scoped_acquire gil;

            // Must make a copy here!
            bool returned = filter_fn(data_object.copy_obj());

            return returned;
        });
    });
}

PythonOperator OperatorsProxy::flatten()
{
    //  Build and return the map operator
    return PythonOperator("flatten", [=](PyObjectObservable source) {
        return rxcpp::observable<>::create<PyHolder>([=](PyObjectSubscriber sink) {
            source.subscribe(
                sink,
                [sink](PyHolder data_object) {
                    try
                    {
                        AcquireGIL gil;

                        // Convert to a vector to allow releasing the GIL
                        std::vector<PyHolder> obj_list;

                        {
                            // Convert to C++ vector while we have the GIL. The list will go out of scope in this block
                            py::list l = py::object(std::move(data_object));

                            for (const auto& item : l)
                            {
                                // This increases the ref count by one but thats fine since the list will go out of
                                // scope and deref all its elements
                                obj_list.emplace_back(std::move(py::reinterpret_borrow<py::object>(item)));
                            }
                        }

                        if (sink.is_subscribed())
                        {
                            // Release the GIL before calling on_next
                            gil.release();

                            // Loop over the list
                            for (auto& i : obj_list)
                            {
                                sink.on_next(std::move(i));
                            }
                        }
                    } catch (py::error_already_set& err)
                    {
                        // Need the GIL here
                        AcquireGIL gil;

                        py::print("Python error in callback hit!");
                        py::print(err.what());

                        // Release before calling on_error
                        gil.release();

                        sink.on_error(std::current_exception());
                    }
                },
                [sink](std::exception_ptr ex) {
                    // Forward
                    sink.on_error(std::move(ex));
                },
                [sink]() {
                    // Forward
                    sink.on_completed();
                });
        });
    });
}

PythonOperator OperatorsProxy::map(std::function<py::object(py::object x)> map_fn)
{
    // Build and return the map operator
    return PythonOperator("map", [=](PyObjectObservable source) -> PyObjectObservable {
        return source.map([=](PyHolder data_object) -> PyHolder {
            py::gil_scoped_acquire gil;

            // Call the map function
            return map_fn(std::move(data_object));
        });
    });
}

PythonOperator OperatorsProxy::on_completed(std::function<py::object()> finally_fn)
{
    return PythonOperator("on_completed", [=](PyObjectObservable source) {
        // Make a new observable
        return rxcpp::observable<>::create<PyHolder>([=](PyObjectSubscriber sink) {
            source.subscribe(rxcpp::make_observer_dynamic<PyHolder>(
                [sink](PyHolder x) {
                    // Forward
                    sink.on_next(std::move(x));
                },
                [sink](std::exception_ptr ex) {
                    // Forward
                    sink.on_error(std::move(ex));
                },
                [sink, finally_fn]() {
                    // In finally function, call the wrapped function
                    auto ret_val = finally_fn();

                    if (ret_val && !ret_val.is_none())
                    {
                        sink.on_next(std::move(ret_val));
                    }

                    // Call on_completed
                    sink.on_completed();
                }));
        });
    });
}

PyHolder wrapper_pair_to_tuple(py::object&& left, py::object&& right)
{
    return py::make_tuple(std::move(left), std::move(right));
}

PythonOperator OperatorsProxy::pairwise()
{
    //  Build and return the map operator
    return PythonOperator("pairwise", [](PyObjectObservable source) {
        return source
            .map([](PyHolder data_object) {
                // py::gil_scoped_acquire gil;
                // Move it into a wrapper in case it goes out of scope
                return PyObjectHolder(std::move(data_object));
            })
            .pairwise()
            .map([](std::tuple<PyObjectHolder, PyObjectHolder> x) {
                // Convert the C++ tuples back into python tuples. Need the GIL since were making a new object
                py::gil_scoped_acquire gil;

                return std::apply(wrapper_pair_to_tuple, std::move(x));
            });
    });
}

template <class T>
struct to_list  // NOLINT
{
    typedef rxcpp::util::decay_t<T> source_value_type;  // NOLINT
    typedef std::vector<source_value_type> value_type;  // NOLINT

    template <class Subscriber>  // NOLINT
    struct to_list_observer      // NOLINT
    {
        typedef to_list_observer<Subscriber> this_type;       // NOLINT
        typedef std::vector<source_value_type> value_type;    // NOLINT
        typedef rxcpp::util::decay_t<Subscriber> dest_type;   // NOLINT
        typedef rxcpp::observer<T, this_type> observer_type;  // NOLINT
        dest_type dest;
        mutable std::vector<source_value_type> remembered;

        to_list_observer(dest_type d) : dest(std::move(d)) {}
        template <typename U>
        void on_next(U&& v) const
        {
            remembered.emplace_back(std::forward<U>(v));
        }
        void on_error(rxcpp::util::error_ptr e) const
        {
            dest.on_error(e);
        }
        void on_completed() const
        {
            if (!remembered.empty())
            {
                dest.on_next(std::move(remembered));
            }

            dest.on_completed();
        }

        static rxcpp::subscriber<T, observer_type> make(dest_type d)
        {
            auto cs = d.get_subscription();
            return rxcpp::make_subscriber<T>(std::move(cs), observer_type(this_type(std::move(d))));
        }
    };

    template <class SubscriberT>
    auto operator()(SubscriberT dest) const -> decltype(to_list_observer<SubscriberT>::make(std::move(dest)))
    {
        return to_list_observer<SubscriberT>::make(std::move(dest));
    }
};

PythonOperator OperatorsProxy::to_list()
{
    //  Build and return the map operator
    return PythonOperator("to_list", [](PyObjectObservable source) {
        using pyobj_to_list_t = ::mrc::pymrc::to_list<PyHolder>;

        // return source.subscribe(sink);
        return source.lift<rxcpp::util::value_type_t<pyobj_to_list_t>>(pyobj_to_list_t())
            .map([](std::vector<PyHolder> obj_list) -> PyHolder {
                AcquireGIL gil;

                // Convert the list back into a python object
                py::list values;

                for (auto& x : obj_list)
                {
                    values.append(py::object(std::move(x)));
                }

                // Clear the list while we still have the GIL
                obj_list.clear();

                return PyHolder(std::move(values));
            });
    });
}
}  // namespace mrc::pymrc
