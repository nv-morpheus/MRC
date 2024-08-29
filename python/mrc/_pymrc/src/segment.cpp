/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>

// IWYU thinks we need array for py::print
// IWYU pragma: no_include <array>

namespace mrc::pymrc {

namespace py = pybind11;

class PyIteratorIterator
{
  public:
    // Iterator traits
    // NOLINTBEGIN(readability-identifier-naming)
    using value_type        = PyHolder;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const value_type*;
    using reference         = const value_type&;
    using iterator_category = std::input_iterator_tag;
    // NOLINTEND(readability-identifier-naming)

    PyIteratorIterator() = default;
    PyIteratorIterator(py::iterator iter) : m_iter(std::move(iter))
    {
        // When creating this object, we want to save the iterator value while we have the GIL. This way we dont have to
        // eagerly grab the GIL even if we already have the value
        DCHECK_EQ(PyGILState_Check(), 1) << "Must have the GIL when creating PyIteratorIterator";

        // This also triggers py::iterator to load the value into memory and avoid needing the GIL unless we are
        // advancing.
        m_value = py::cast<py::object>(*m_iter);
    }

    ~PyIteratorIterator()
    {
        if (m_iter)
        {
            AcquireGIL gil;

            py::iterator kill = std::move(m_iter);
            m_value           = py::object();
        }
    }

    bool is_initialized() const
    {
        return static_cast<bool>(m_iter);
    }

    // Pre increment
    PyIteratorIterator& operator++()
    {
        advance();
        return *this;
    }

    // Post increment
    PyIteratorIterator operator++(int)
    {
        auto rv = *this;
        advance();
        return rv;
    }

    reference operator*() const
    {
        return m_value;
    }

    pointer operator->() const
    {
        this->operator*();
        return &m_value;
    }

    friend bool operator==(const PyIteratorIterator& a, const PyIteratorIterator& b)
    {
        // Define as the opposite. operator!= is used directly more often
        return !(a != b);
    }

    friend bool operator!=(const PyIteratorIterator& a, const PyIteratorIterator& b)
    {
        // If we use a.m_iter != b.m_iter, the iterators will compare the values by dereferencing the iterator. This may
        // require the GIL. Since we already have the value, just directly compare that instead. This results in the
        // same check being performed without requiring the GIL
        return a.m_value.ptr() != b.m_value.ptr();
    }

  private:
    void advance()
    {
        // Grab the GIL before advancing
        AcquireGIL gil;

        ++m_iter;

        // While we have the GIL, preload the next value
        m_value = py::cast<py::object>(*m_iter);
    }

    py::iterator m_iter{};
    PyHolder m_value{};
};

class PyIteratorWrapper
{
  public:
    // NOLINTBEGIN(readability-identifier-naming)
    using iterator = PyIteratorIterator;
    // NOLINTEND(readability-identifier-naming)

    // Create from an iterator
    PyIteratorWrapper(py::iterator source_iterator) :
      m_iter_factory([iterator = PyObjectHolder(std::move(source_iterator))]() mutable {
          // Check if the iterator has been started already
          if (!iterator)
          {
              LOG(ERROR) << "Cannot have multiple progress engines for the iterator overload. Iterators cannot be "
                            "duplicated";
              throw std::runtime_error(
                  "Cannot have multiple progress engines for the iterator overload. Iterators cannot be duplicated");
          }

          // Move the object into the iterator to ensure its only used once.
          return py::cast<py::iterator>(py::object(std::move(iterator)));
      })
    {}

    // Create from an iterable
    PyIteratorWrapper(py::iterable source_iterable) :
      m_iter_factory([iterable = PyObjectHolder(std::move(source_iterable))]() {
          // Turn the iterable into an iterator
          return py::iter(iterable);
      })
    {}

    // Create from a factory function
    PyIteratorWrapper(py::function gen_factory) :
      m_iter_factory([gen_factory = PyObjectHolder(std::move(gen_factory))]() {
          // Call the generator factory to make a new generator
          return py::cast<py::iterator>(gen_factory());
      })
    {}

    // Create directly
    PyIteratorWrapper(std::function<py::iterator()> iter_factory) : m_iter_factory(std::move(iter_factory)) {}

    iterator begin()
    {
        // Grab the GIL and create the iterator from the factory
        AcquireGIL gil;

        auto iter = m_iter_factory();

        return iterator{std::move(iter)};
    }

    iterator end()  // NOLINT(readability-convert-member-functions-to-static)
    {
        // Do we need the GIL here?
        return iterator{};
    }

  private:
    std::function<py::iterator()> m_iter_factory;
};

std::shared_ptr<mrc::segment::ObjectProperties> build_source(mrc::segment::IBuilder& self,
                                                             const std::string& name,
                                                             PyIteratorWrapper iter_wrapper)
{
    auto wrapper = [iter_wrapper = std::move(iter_wrapper)](PyObjectSubscriber& subscriber) mutable {
        auto& ctx = runnable::Context::get_runtime_context();

        try
        {
            DVLOG(10) << ctx.info() << " Starting source";

            for (auto next_val : iter_wrapper)
            {
                //  Only send if its subscribed. Very important to ensure the object has been moved!
                if (subscriber.is_subscribed())
                {
                    subscriber.on_next(std::move(next_val));
                }
                else
                {
                    DVLOG(10) << ctx.info() << " Source unsubscribed. Stopping";
                    break;
                }
            }

        } catch (const std::exception& e)
        {
            LOG(ERROR) << ctx.info() << "Error occurred in source. Error msg: " << e.what();

            // gil.release();
            subscriber.on_error(std::current_exception());
            return;
        }

        subscriber.on_completed();

        DVLOG(10) << ctx.info() << " Source complete";
    };

    return self.construct_object<PythonSource<PyHolder>>(name, wrapper);
}

std::shared_ptr<mrc::segment::ObjectProperties> build_source_component(mrc::segment::IBuilder& self,
                                                                       const std::string& name,
                                                                       PyIteratorWrapper iter_wrapper)
{
    auto get_next = [iter_wrapper = std::move(iter_wrapper), current = PyIteratorIterator()](PyHolder& data) mutable {
        // // Unfortunately, all of the below steps need the GIL but will grab it individually. To prevent that grab
        // // the GIL now to avoid dropping and reacquiring multiple times
        // AcquireGIL gil;

        if (!current.is_initialized())
        {
            // On the first pass, initialize the iterator. This will generate the first value
            current = iter_wrapper.begin();
        }
        else
        {
            // Otherwise pre-increment the iterator
            ++current;
        }

        // Copy the current value into data. No need for the GIL here due to PyHolder
        data = *current;

        // Return closed if the current iterator is complete
        return current == PyIteratorIterator() ? channel::Status::closed : channel::Status::success;
    };

    return self.construct_object<PythonSourceComponent<PyHolder>>(name, std::move(get_next));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::IBuilder& self,
                                                                          const std::string& name,
                                                                          py::iterator source_iterator)
{
    return build_source(self, name, PyIteratorWrapper(std::move(source_iterator)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::IBuilder& self,
                                                                          const std::string& name,
                                                                          py::iterable source_iterable)
{
    return build_source(self, name, PyIteratorWrapper(std::move(source_iterable)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source(mrc::segment::IBuilder& self,
                                                                          const std::string& name,
                                                                          py::function gen_factory)
{
    return build_source(self, name, PyIteratorWrapper(std::move(gen_factory)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source_component(mrc::segment::IBuilder& self,
                                                                                    const std::string& name,
                                                                                    pybind11::iterator source_iterator)
{
    return build_source_component(self, name, PyIteratorWrapper(std::move(source_iterator)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source_component(mrc::segment::IBuilder& self,
                                                                                    const std::string& name,
                                                                                    py::iterable source_iterable)
{
    return build_source_component(self, name, PyIteratorWrapper(std::move(source_iterable)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_source_component(mrc::segment::IBuilder& self,
                                                                                    const std::string& name,
                                                                                    pybind11::function gen_factory)
{
    return build_source_component(self, name, PyIteratorWrapper(std::move(gen_factory)));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_sink(mrc::segment::IBuilder& self,
                                                                        const std::string& name,
                                                                        OnNextFunction on_next,
                                                                        OnErrorFunction on_error,
                                                                        OnCompleteFunction on_completed)
{
    return self.make_sink<PyHolder, PythonSink>(name, on_next, on_error, on_completed);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_sink_component(mrc::segment::IBuilder& self,
                                                                                  const std::string& name,
                                                                                  OnNextFunction on_next,
                                                                                  OnErrorFunction on_error,
                                                                                  OnCompleteFunction on_completed)
{
    return self.make_sink_component<PyHolder, PythonSinkComponent>(name, on_next, on_error, on_completed);
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::get_ingress(mrc::segment::IBuilder& self,
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

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::get_egress(mrc::segment::IBuilder& self,
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

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node(mrc::segment::IBuilder& self,
                                                                        const std::string& name,
                                                                        OnDataFunction on_data)
{
    show_deprecation_warning(
        "Passing a map function object to make_node() is deprecated and will be removed in a future version. "
        "make_node() now requires an operator. Use "
        "make_node(name, mrc.core.operators.map(map_fn)) instead.");

    return BuilderProxy::make_node(self, name, py::args(py::make_tuple(py::cast(OperatorsProxy::map(on_data)))));
}

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node(mrc::segment::IBuilder& self,
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
    mrc::segment::IBuilder& self,
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

std::shared_ptr<mrc::segment::ObjectProperties> BuilderProxy::make_node_component(mrc::segment::IBuilder& self,
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
    mrc::segment::IBuilder& self,
    const std::string& module_id,
    const std::string& registry_namespace,
    std::string module_name,
    py::dict config)
{
    auto json_config = cast_from_pyobject(config);

    return self.load_module_from_registry(module_id, registry_namespace, std::move(module_name), std::move(json_config));
}

void BuilderProxy::init_module(mrc::segment::IBuilder& self, std::shared_ptr<mrc::modules::SegmentModule> smodule)
{
    self.init_module(smodule);
}

void BuilderProxy::register_module_input(mrc::segment::IBuilder& self,
                                         std::string input_name,
                                         std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_input(std::move(input_name), object);
}

void BuilderProxy::register_module_output(mrc::segment::IBuilder& self,
                                          std::string output_name,
                                          std::shared_ptr<segment::ObjectProperties> object)
{
    self.register_module_output(std::move(output_name), object);
}

py::dict BuilderProxy::get_current_module_config(mrc::segment::IBuilder& self)
{
    auto json_config = self.get_current_module_config();

    return cast_from_json(json_config);
}

void BuilderProxy::make_edge(mrc::segment::IBuilder& self,
                             std::shared_ptr<mrc::segment::ObjectProperties> source,
                             std::shared_ptr<mrc::segment::ObjectProperties> sink)
{
    if (source->is_writable_acceptor() && sink->is_writable_provider())
    {
        mrc::make_edge_typeless(source->writable_acceptor_base(), sink->writable_provider_base());
    }
    else if (source->is_readable_provider() && sink->is_readable_acceptor())
    {
        mrc::make_edge_typeless(source->readable_provider_base(), sink->readable_acceptor_base());
    }
    else
    {
        throw std::runtime_error(
            "Invalid edges. Arguments to make_edge were incorrect. Ensure you are providing either "
            "WritableAcceptor->WritableProvider or ReadableProvider->ReadableAcceptor");
    }
}

}  // namespace mrc::pymrc
