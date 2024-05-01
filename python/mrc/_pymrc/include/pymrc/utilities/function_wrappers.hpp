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

#include "pymrc/utilities/object_wrappers.hpp"  // IWYU pragma: keep

#include "mrc/utils/string_utils.hpp"

#include <pybind11/cast.h>
#include <pybind11/detail/descr.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

struct PyFuncWrapper
{
  public:
    PyFuncWrapper() = default;
    explicit PyFuncWrapper(pybind11::function&& fn);

    ~PyFuncWrapper();

    PyFuncWrapper(const PyFuncWrapper& other);

    PyFuncWrapper& operator=(const PyFuncWrapper& other);

    template <typename ReturnT, typename... ArgsT>
    ReturnT operator()(ArgsT... args) const
    {
        pybind11::gil_scoped_acquire acq;

        pybind11::object retval(m_fn(std::forward<ArgsT>(args)...));

        /* Visual studio 2015 parser issue: need parentheses around this expression */
        return (retval.template cast<ReturnT>());
    }

    const pybind11::function& py_function_obj() const;

    pybind11::function& py_function_obj();

    const std::string& repr() const;

  private:
    pybind11::function m_fn;
    std::string m_repr;
};

template <typename SignatureT>
struct PyFuncHolder;

template <typename ReturnT, typename... ArgsT>
struct PyFuncHolder<ReturnT(ArgsT...)>
{
  public:
    using cpp_fn_t       = std::function<ReturnT(ArgsT...)>;
    using return_t       = std::conditional_t<std::is_same<ReturnT, void>::value, pybind11::detail::void_type, ReturnT>;
    using function_ptr_t = ReturnT (*)(ArgsT...);

    // Default construct with an empty object. Needed by pybind11 casters
    PyFuncHolder() = default;

    // Object is default copyable and moveable
    PyFuncHolder(const PyFuncHolder&)            = default;
    PyFuncHolder& operator=(const PyFuncHolder&) = default;
    PyFuncHolder(PyFuncHolder&&)                 = default;
    PyFuncHolder& operator=(PyFuncHolder&&)      = default;

    ReturnT operator()(ArgsT... args) const
    {
        // Throw an error if we have not been initialized
        if (!m_cpp_fn)
        {
            throw std::runtime_error("Cannot call python wrapped function that has not been initialized");
        }

        return m_cpp_fn(std::forward<ArgsT>(args)...);
    }

    static constexpr auto Signature = pybind11::detail::_("Callable[[") +
                                      pybind11::detail::concat(pybind11::detail::make_caster<ArgsT>::name...) +
                                      pybind11::detail::_("], ") + pybind11::detail::make_caster<return_t>::name +
                                      pybind11::detail::_("]");

    // We require a factory function to get around calling virtual functions inside of the constructor. These objects
    // are default constructable but will throw an error
    template <typename DerivedT>
    static DerivedT create(pybind11::function&& fn)
    {
        // Default construct the object
        DerivedT derived{};

        // Set the function
        derived.set_py_function(std::move(fn));

        return derived;
    }

  protected:
    virtual cpp_fn_t build_cpp_function(pybind11::function&& py_fn) const
    {
        // Check if the object is None
        if (!py_fn)
        {
            // If the return type is void, we can make a default function
            if constexpr (std::is_void_v<ReturnT>)
            {
                return [](ArgsT... inner_args) -> void {};
            }
            else
            {
                throw std::runtime_error(MRC_CONCAT_STR("Python function argument '"
                                                        << std::string(pybind11::str(py_fn))
                                                        << "', cannot be None since it returns a value"));
            }
        }

        // Default implementation just forwards to the holder
        return [holder = PyFuncWrapper(std::move(py_fn))](ArgsT... inner_args) -> ReturnT {
            return holder.operator()<ReturnT, ArgsT...>(std::forward<ArgsT>(inner_args)...);
        };
    }

  private:
    void set_py_function(pybind11::function&& py_fn)
    {
        // Save the name of the function to help debugging
        if (py_fn)
        {
            m_repr = pybind11::str(py_fn);
        }

        m_cpp_fn = this->build_cpp_function(std::move(py_fn));
    }

    cpp_fn_t m_cpp_fn;
    std::string m_repr;
};

struct OnNextFunction : public PyFuncHolder<void(PyObjectHolder)>
{
  public:
    using base_t = PyFuncHolder<void(PyObjectHolder)>;

    OnNextFunction() = default;

    static constexpr auto Signature = pybind11::detail::_("Callable[[object], None]");

  protected:
    using base_t::cpp_fn_t;

    cpp_fn_t build_cpp_function(pybind11::function&& py_fn) const override;
};

struct OnErrorFunction : public PyFuncHolder<void(std::exception_ptr)>
{
  public:
    using base_t = PyFuncHolder<void(std::exception_ptr)>;

    OnErrorFunction() = default;

    static constexpr auto Signature = pybind11::detail::_("Callable[[BaseException], None]");

  protected:
    using base_t::cpp_fn_t;

    cpp_fn_t build_cpp_function(pybind11::function&& py_fn) const override;
};

struct OnCompleteFunction : public PyFuncHolder<void()>
{
  public:
    using base_t = PyFuncHolder<void()>;

    OnCompleteFunction() = default;

    static constexpr auto Signature = pybind11::detail::_("Callable[[], None]");

  protected:
    using base_t::cpp_fn_t;
};

// OnDataFunction, like other lambdas for operators, takes normal pybind11::object. This is because these functions will
// be interfacing directly with user python code and is never holding the objects in memory after the call is made
struct OnDataFunction : public PyFuncHolder<pybind11::object(pybind11::object)>
{
  public:
    using base_t = PyFuncHolder<pybind11::object(pybind11::object)>;

    OnDataFunction() = default;

    static constexpr auto Signature = pybind11::detail::_("Callable[[object], object]");

  protected:
    using base_t::cpp_fn_t;

    cpp_fn_t build_cpp_function(pybind11::function&& py_fn) const override;
};

#pragma GCC visibility pop

}  // namespace mrc::pymrc

namespace pybind11::detail {

template <typename DerivedWrapperT>
class PyFuncWrapperCasterBase
{
    using type       = DerivedWrapperT;
    using retval_t   = typename DerivedWrapperT::return_t;
    using function_t = typename DerivedWrapperT::function_ptr_t;

  public:
    bool load(handle src, bool convert)
    {
        if (src.is_none())
        {
            // Defer accepting None to other overloads (if we aren't in convert mode)
            if (!convert)
            {
                return false;
            }

            // Otherwise, build an instance from None. This is different than the default constructed value which will
            // throw an error when used
            value = DerivedWrapperT::template create<DerivedWrapperT>(pybind11::function());

            return true;
        }

        if (!isinstance<function>(src))
        {
            return false;
        }

        auto func = reinterpret_borrow<function>(src);

        // /*
        //    When passing a C++ function as an argument to another C++
        //    function via Python, every function call would normally involve
        //    a full C++ -> Python -> C++ roundtrip, which can be prohibitive.
        //    Here, we try to at least detect the case where the function is
        //    stateless (i.e. function pointer or lambda function without
        //    captured variables), in which case the roundtrip can be avoided.
        //  */
        // if (auto cfunc = func.cpp_function())
        // {
        //     auto cfunc_self = PyCFunction_GET_SELF(cfunc.ptr());
        //     if (isinstance<capsule>(cfunc_self))
        //     {
        //         auto c   = reinterpret_borrow<capsule>(cfunc_self);
        //         auto rec = (function_record*)c;

        //         while (rec != nullptr)
        //         {
        //             if (rec->is_stateless &&
        //                 same_type(typeid(function_type), *reinterpret_cast<const std::type_info*>(rec->data[1])))
        //             {
        //                 struct capture
        //                 {
        //                     function_type f;
        //                 };
        //                 value = ((capture*)&rec->data)->f;
        //                 return true;
        //             }
        //             rec = rec->next;
        //         }
        //     }
        //     // PYPY segfaults here when passing builtin function like sum.
        //     // Raising an fail exception here works to prevent the segfault, but only on gcc.
        //     // See PR #1413 for full details
        // }

        //         // ensure GIL is held during functor destruction
        //         struct func_handle
        //         {
        //             function f;
        // #if !(defined(_MSC_VER) && _MSC_VER == 1916 && defined(PYBIND11_CPP17))
        //             // This triggers a syntax error under very special conditions (very weird indeed).
        //             explicit
        // #endif
        //                 func_handle(function&& f_) noexcept :
        //               f(std::move(f_))
        //             {}
        //             func_handle(const func_handle& f_)
        //             {
        //                 operator=(f_);
        //             }
        //             func_handle& operator=(const func_handle& f_)
        //             {
        //                 gil_scoped_acquire acq;
        //                 f = f_.f;
        //                 return *this;
        //             }
        //             ~func_handle()
        //             {
        //                 gil_scoped_acquire acq;
        //                 function kill_f(std::move(f));
        //             }
        //         };

        //         // to emulate 'move initialization capture' in C++11
        //         struct func_wrapper
        //         {
        //             func_handle hfunc;
        //             explicit func_wrapper(func_handle&& hf) noexcept : hfunc(std::move(hf)) {}
        //             void operator()(mrc::pymrc::PyObjectHolder args) const
        //             {
        //                 gil_scoped_acquire acq;
        //                 object retval(hfunc.f(std::move(args)));
        //                 /* Visual studio 2015 parser issue: need parentheses around this expression */
        //                 // return (retval.template cast<void>());
        //             }
        //         };

        // Use the factory function to create with a function object
        value = DerivedWrapperT::template create<DerivedWrapperT>(std::move(func));

        return true;
    }

    template <typename FuncT>
    static handle cast(FuncT&& f, return_value_policy policy, handle /* parent */)
    {
        if (!f)
        {
            return none().inc_ref();
        }

        auto result = f.template target<function_t>();

        if (result)
        {
            return cpp_function(*result, policy).release();
        }

        return cpp_function(std::forward<FuncT>(f), policy).release();
    }

    PYBIND11_TYPE_CASTER(type, DerivedWrapperT::Signature);
};

template <typename ReturnT, typename... ArgsT>
class type_caster<mrc::pymrc::PyFuncHolder<ReturnT(ArgsT...)>>
  : public PyFuncWrapperCasterBase<mrc::pymrc::PyFuncHolder<ReturnT(ArgsT...)>>
{};

template <>
class type_caster<mrc::pymrc::OnNextFunction> : public PyFuncWrapperCasterBase<mrc::pymrc::OnNextFunction>
{};

template <>
class type_caster<mrc::pymrc::OnErrorFunction> : public PyFuncWrapperCasterBase<mrc::pymrc::OnErrorFunction>
{};

template <>
class type_caster<mrc::pymrc::OnCompleteFunction> : public PyFuncWrapperCasterBase<mrc::pymrc::OnCompleteFunction>
{};

template <>
class type_caster<mrc::pymrc::OnDataFunction> : public PyFuncWrapperCasterBase<mrc::pymrc::OnDataFunction>
{};
}  // namespace pybind11::detail
