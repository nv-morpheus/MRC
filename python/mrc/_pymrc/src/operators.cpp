/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "pymrc/utilities/acquire_gil.hpp"
#include "pymrc/utilities/function_wrappers.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU pragma: no_include <array>

namespace mrc::pymrc {

namespace py = pybind11;

PythonOperator::PythonOperator(std::string name, PyObjectOperateFn operate_fn) :
  m_name(std::move(name)),
  m_operate_fn(std::move(operate_fn))
{}
const std::string& PythonOperator::get_name() const
{
    return m_name;
}
const PyObjectOperateFn& PythonOperator::get_operate_fn() const
{
    return m_operate_fn;
}

std::string OperatorProxy::get_name(PythonOperator& self)
{
    return self.get_name();
}

PythonOperator OperatorsProxy::build(PyFuncHolder<void(const PyObjectObservable& obs, PyObjectSubscriber& sub)> build_fn)
{
    //  Build and return the map operator
    return {"build", [=](PyObjectObservable source) -> PyObjectObservable {
                return rxcpp::observable<>::create<PyHolder>([source, build_fn](pymrc::PyObjectSubscriber output) {
                    try
                    {
                        py::gil_scoped_acquire gil;

                        // Call the subscribe function
                        build_fn(source, output);

                        return output;

                    } catch (py::error_already_set& err)
                    {
                        LOG(ERROR)
                            << "Python occurred during full node subscription. Error: " + std::string(err.what());

                        // Rethrow python exceptions
                        throw;
                    } catch (std::exception& err)
                    {
                        LOG(ERROR) << "Exception occurred during subscription. Error: " + std::string(err.what());
                        throw;
                    }
                });
            }};
}

PythonOperator OperatorsProxy::filter(PyFuncHolder<bool(pybind11::object x)> filter_fn)
{
    //  Build and return the map operator
    return {"filter", [=](PyObjectObservable source) {
                return source.filter([=](PyHolder data_object) {
                    py::gil_scoped_acquire gil;

                    // Must make a copy here!
                    bool returned = filter_fn(data_object.copy_obj());

                    return returned;
                });
            }};
}

PythonOperator OperatorsProxy::flatten()
{
    //  Build and return the map operator
    return {"flatten", [=](PyObjectObservable source) {
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
                                    // Convert to C++ vector while we have the GIL. The list will go out of scope in
                                    // this block
                                    py::list l = py::object(std::move(data_object));

                                    for (const auto& item : l)
                                    {
                                        // This increases the ref count by one but thats fine since the list will go out
                                        // of scope and deref all its elements
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
            }};
}

PythonOperator OperatorsProxy::flatmap(PyFuncHolder<PyObjectHolder(pybind11::object)> flatmap_fn)
{
    //  Build and return the map operator
    return {"flatten", [=](PyObjectObservable source) {
                return rxcpp::observable<>::create<PyHolder>([=](PyObjectSubscriber sink) {
                    AcquireGIL gil;

                    auto asyncio = pybind11::module_::import("asyncio");
                    PyHolder loop;

                    try
                    {
                        loop = asyncio.attr("get_event_loop")();
                    } catch (std::runtime_error ex)
                    {
                        loop = asyncio.attr("new_event_loop")();
                    }

                    auto is_running = loop.attr("is_running")().cast<bool>();

                    std::thread loop_thread;

                    std::atomic<bool> cancellation_token = false;

                    if (not is_running)
                    {
                        loop_thread = std::thread([&cancellation_token, loop = loop]() {
                            while (not cancellation_token)
                            {
                                {
                                    // run event loop once
                                    pybind11::gil_scoped_acquire acquire;
                                    loop.attr("stop")();
                                    loop.attr("run_forever")();
                                }
                                std::this_thread::yield();
                            }
                        });
                    }

                    gil.release();

                    source.subscribe(
                        sink,
                        [sink, flatmap_fn, loop](PyHolder value) {
                            try
                            {
                                AcquireGIL gil;

                                auto result = flatmap_fn(std::move(value));

                                auto inspect = pybind11::module_::import("inspect");

                                if (inspect.attr("iscoroutine")(result).cast<bool>())
                                {
                                    gil.release();
                                    throw std::runtime_error("flatmap does not yet support coroutines");
                                }

                                if (inspect.attr("isasyncgen")(result).cast<bool>())
                                {
                                    auto asyncio = py::module_::import("asyncio");

                                    while (true)
                                    {
                                        auto task   = result.attr("__anext__")();
                                        auto future = asyncio.attr("run_coroutine_threadsafe")(task, loop);

                                        while (not future.attr("done")().cast<bool>()) {
                                            using namespace std::chrono_literals;
                                            pybind11::gil_scoped_release release;
                                            std::this_thread::yield();
                                            // std::this_thread::sleep_for(100ms);
                                        }

                                        try{
                                            auto value = future.attr("result")();
                                            pybind11::gil_scoped_release release;
                                            sink.on_next(std::move(py::reinterpret_borrow<py::object>(value)));
                                        } catch (std::exception ex){
                                            // probably an end async iteration exception.
                                            break;
                                        }
                                    }

                                    return;
                                }

                                std::vector<PyHolder> obj_list;

                                {
                                    auto l = py::list(std::move(result));

                                    for (const auto& item : l)
                                    {
                                        // This increases the ref count by one but thats fine since the list will go
                                        // out of scope and deref all its elements
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
                            // should actually wait for all futures to be completed
                            sink.on_completed();
                        });

                    if (loop_thread.joinable())
                    {
                        cancellation_token = true;
                        loop_thread.join();
                    }
                });
            }};
}

PythonOperator OperatorsProxy::map(OnDataFunction map_fn)
{
    // Build and return the map operator
    return {"map", [=](PyObjectObservable source) -> PyObjectObservable {
                return source.map([=](PyHolder data_object) -> PyHolder {
                    py::gil_scoped_acquire gil;

                    // Call the map function
                    return map_fn(std::move(data_object));
                });
            }};
}

PythonOperator OperatorsProxy::on_completed(PyFuncHolder<std::optional<pybind11::object>()> finally_fn)
{
    return {"on_completed", [=](PyObjectObservable source) {
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

                            if (ret_val.has_value() && !ret_val.value().is_none())
                            {
                                sink.on_next(std::move(ret_val.value()));
                            }

                            // Call on_completed
                            sink.on_completed();
                        }));
                });
            }};
}

PyHolder wrapper_pair_to_tuple(py::object&& left, py::object&& right)
{
    return py::make_tuple(std::move(left), std::move(right));
}

PythonOperator OperatorsProxy::pairwise()
{
    //  Build and return the map operator
    return {"pairwise", [](PyObjectObservable source) {
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
            }};
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
    return {"to_list", [](PyObjectObservable source) {
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

                        return {std::move(values)};
                    });
            }};
}

}  // namespace mrc::pymrc
