/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../test_mrc.hpp"  // IWYU pragma: associated
#include "../test_nodes.hpp"

#include "mrc/exceptions/runtime_error.hpp"  // for MrcRuntimeError
#include "mrc/node/operators/router.hpp"
#include "mrc/type_traits.hpp"

#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <memory>
#include <type_traits>
#include <vector>

TEST_CLASS(Router);
namespace {
template <typename T>
std::string even_odd(const T& t)
{
    return t % 2 == 1 ? "odd" : "even";
}

template <typename T>
int mod_three(const T& data)
{
    return data % 3;
}
};  // namespace

namespace mrc::node {

template <typename T, typename BaseT>
class DerivedRouterBase : public BaseT
{
  public:
    using this_t = DerivedRouterBase<T, BaseT>;
    using BaseT::BaseT;

    void run()
    {
        constexpr bool has_do_run = requires(this_t& t) { t.do_run(); };

        if constexpr (has_do_run)
        {
            this->do_run();
        }
    }

  protected:
    int determine_key_for_value(const T& t) override
    {
        return mod_three(t);
    }
};

template <typename BaseT>
class DerivedLambdaRouter : public BaseT
{
  public:
    using this_t = DerivedLambdaRouter<BaseT>;
    using BaseT::BaseT;

    void run()
    {
        constexpr bool has_do_run = requires(this_t& t) { t.do_run(); };

        if constexpr (has_do_run)
        {
            this->do_run();
        }
    }
};

// template <typename T>
// class TestStaticRouterComponent : public StaticRouterComponentBase2<int, T>
// {
//   public:
//     using base_t = StaticRouterComponentBase2<int, T>;

//     TestStaticRouterComponent() : base_t(std::vector<int>{0, 1, 2}) {}

//   protected:
//     int determine_key_for_value(const T& t) override
//     {
//         return mod_three(t);
//     }
// };

// template <typename T>
// class TestStaticRouterRunnable : public StaticRouterRunnableBase2<int, T>
// {
//   public:
//     using base_t = StaticRouterRunnableBase2<int, T>;

//     TestStaticRouterRunnable() : base_t(std::vector<int>{0, 1, 2}) {}

//     void run()
//     {
//         this->do_run();
//     }

//   protected:
//     int determine_key_for_value(const T& t) override
//     {
//         return mod_three(t);
//     }
// };

// template <typename T>
// class TestDynamicRouterComponent : public DynamicRouterComponentBase2<int, T>
// {
//   public:
//     using base_t = DynamicRouterComponentBase2<int, T>;

//     TestDynamicRouterComponent() : base_t() {}

//   protected:
//     int determine_key_for_value(const T& t) override
//     {
//         return mod_three(t);
//     }
// };

// template <typename T>
// class TestDynamicRouterRunnable : public DynamicRouterRunnableBase2<int, T>
// {
//   public:
//     using base_t = DynamicRouterRunnableBase2<int, T>;

//     TestDynamicRouterRunnable() : base_t() {}

//     void run()
//     {
//         this->do_run();
//     }

//   protected:
//     int determine_key_for_value(const T& t) override
//     {
//         return mod_three(t);
//     }
// };

template <typename T>
class TestStaticRouterComponentString : public StaticRouterComponentBase<std::string, T>
{
  public:
    using base_t = StaticRouterComponentBase<std::string, T>;

    TestStaticRouterComponentString() : base_t(std::vector<std::string>{"odd", "even"}) {}

  protected:
    std::string determine_key_for_value(const T& t) override
    {
        return even_odd(t);
    }
};

}  // namespace mrc::node

namespace mrc {

template <typename T>
    requires is_base_of_template_v<node::RouterBase, T>
class TestRouterTypes : public testing::Test
{
  public:
    void SetUp() override
    {
        auto source = std::make_shared<node::TestSource<int>>(10);
        auto router = this->create_router();
        this->sink0 = std::make_shared<node::TestSink<int>>();
        this->sink1 = std::make_shared<node::TestSink<int>>();
        this->sink2 = std::make_shared<node::TestSink<int>>();

        mrc::make_edge(*source, *router);
        mrc::make_edge(*router->get_source(0), *this->sink0);
        mrc::make_edge(*router->get_source(1), *this->sink1);
        mrc::make_edge(*router->get_source(2), *this->sink2);

        constexpr bool has_run = requires(T& t) { t.run(); };

        source->run();

        // If it has the run method, then call that
        if constexpr (has_run)
        {
            router->run();
        }

        this->sink0->run();
        this->sink1->run();
        this->sink2->run();
    }

  protected:
    std::shared_ptr<T> create_router()
    {
        if constexpr (std::is_constructible_v<T, std::vector<int>, std::function<int(const int&)>>)
        {
            return std::make_shared<T>(std::vector<int>{0, 1, 2}, [](const int& data) {
                return mod_three(data);
            });
        }
        else if constexpr (std::is_constructible_v<T, std::vector<int>>)
        {
            return std::make_shared<T>(std::vector<int>{0, 1, 2});
        }
        else if constexpr (std::is_constructible_v<T, std::function<int(const int&)>>)
        {
            return std::make_shared<T>([](const int& data) {
                return mod_three(data);
            });
        }
        else if constexpr (std::is_default_constructible_v<T>)
        {
            return std::make_shared<T>();
        }
        else
        {
            static_assert(!sizeof(T), "Unsupported router type");
        }
    }

    std::shared_ptr<node::TestSink<int>> sink0;
    std::shared_ptr<node::TestSink<int>> sink1;
    std::shared_ptr<node::TestSink<int>> sink2;
};

using RouterTypes = ::testing::Types<node::DerivedRouterBase<int, node::StaticRouterComponentBase<int, int>>,
                                     node::DerivedRouterBase<int, node::StaticRouterRunnableBase<int, int>>,
                                     node::DerivedRouterBase<int, node::DynamicRouterComponentBase<int, int>>,
                                     node::DerivedRouterBase<int, node::DynamicRouterRunnableBase<int, int>>,
                                     node::DerivedLambdaRouter<node::LambdaStaticRouterComponent<int, int>>,
                                     node::DerivedLambdaRouter<node::LambdaStaticRouterRunnable<int, int>>,
                                     node::DerivedLambdaRouter<node::LambdaDynamicRouterComponent<int, int>>,
                                     node::DerivedLambdaRouter<node::LambdaDynamicRouterRunnable<int, int>>>;

class RouterTypesNameGenerator
{
  public:
    template <typename T>
    static std::string GetName(int)
    {
        if constexpr (std::is_same_v<T, node::DerivedRouterBase<int, node::StaticRouterComponentBase<int, int>>>)
            return "StaticComponentBase";
        if constexpr (std::is_same_v<T, node::DerivedRouterBase<int, node::StaticRouterRunnableBase<int, int>>>)
            return "StaticRunnableBase";
        if constexpr (std::is_same_v<T, node::DerivedRouterBase<int, node::DynamicRouterComponentBase<int, int>>>)
            return "DynamicComponentBase";
        if constexpr (std::is_same_v<T, node::DerivedRouterBase<int, node::DynamicRouterRunnableBase<int, int>>>)
            return "DynamicRunnableBase";
        if constexpr (std::is_same_v<T, node::DerivedLambdaRouter<node::LambdaStaticRouterComponent<int, int>>>)
            return "LambdaStaticComponent";
        if constexpr (std::is_same_v<T, node::DerivedLambdaRouter<node::LambdaStaticRouterRunnable<int, int>>>)
            return "LambdaStaticRunnable";
        if constexpr (std::is_same_v<T, node::DerivedLambdaRouter<node::LambdaDynamicRouterComponent<int, int>>>)
            return "LambdaDynamicComponent";
        if constexpr (std::is_same_v<T, node::DerivedLambdaRouter<node::LambdaDynamicRouterRunnable<int, int>>>)
            return "LambdaDynamicRunnable";
    }
};

TYPED_TEST_SUITE(TestRouterTypes, RouterTypes, RouterTypesNameGenerator);

TYPED_TEST(TestRouterTypes, CorrectValues)
{
    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), this->sink0->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), this->sink1->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), this->sink2->get_values());
}

TYPED_TEST(TestRouterTypes, NonExistentSource)
{
    auto router = this->create_router();

    EXPECT_THROW({ router->get_source(-1); }, exceptions::MrcRuntimeError);
}

TEST_F(TestRouter, StaticRouterComponent_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestStaticRouterComponentString<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, StaticRouterComponent_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestStaticRouterComponentString<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, LambdaStaticRouterComponent_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>(10);
    auto router = std::make_shared<node::LambdaStaticRouterComponent<int, int>>(std::vector<int>{1, 2, 3},
                                                                                [](const int& data) {
                                                                                    return (data % 3) + 1;
                                                                                });
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();
    auto sink3  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);
    mrc::make_edge(*router->get_source(3), *sink3);

    source->run();
    sink1->run();
    sink2->run();
    sink3->run();

    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), sink2->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), sink3->get_values());
}

TEST_F(TestRouter, LambdaStaticRouterComponent_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>(10);
    auto router = std::make_shared<node::LambdaStaticRouterComponent<int, int>>(std::vector<int>{1, 2, 3},
                                                                                [](const int& data) {
                                                                                    return (data % 3) + 1;
                                                                                });
    auto sink1  = std::make_shared<node::TestSinkComponent<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();
    auto sink3  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);
    mrc::make_edge(*router->get_source(3), *sink3);

    source->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), sink2->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), sink3->get_values());
}

TEST_F(TestRouter, LambdaRouter_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::LambdaDynamicRouterComponent<std::string, int>>(&even_odd<int>);
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, LambdaRouter_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::LambdaDynamicRouterComponent<std::string, int>>(&even_odd<int>);
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

// two possible errors, either the key function throws an error, or the key function returns a key that is invalid
TEST_F(TestRouter, LambdaRouterOnKeyError)
{
    auto source = std::make_shared<node::TestSource<int>>(3);
    auto router = std::make_shared<node::LambdaDynamicRouterComponent<int, int>>([](const int& data) {
        if (data == 2)
        {
            throw std::runtime_error("Test Error");
        }

        return data + 1;
    });

    auto sink1 = std::make_shared<node::TestSink<int>>();
    auto sink2 = std::make_shared<node::TestSink<int>>();
    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);

    EXPECT_THROW(
        {
            source->run();
            sink2->run();
            sink2->run();
        },
        exceptions::MrcRuntimeError);
}

TEST_F(TestRouter, LambdaRouterOnKeyInvalidValue)
{
    auto source = std::make_shared<node::TestSource<int>>(3);
    auto router = std::make_shared<node::LambdaDynamicRouterComponent<int, int>>([](const int& data) {
        return data + 1;  // On the third value, this will return 4, which is not a valid key
    });

    auto sink1 = std::make_shared<node::TestSink<int>>();
    auto sink2 = std::make_shared<node::TestSink<int>>();
    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);

    EXPECT_THROW(
        {
            source->run();
            sink2->run();
            sink2->run();
        },
        exceptions::MrcRuntimeError);
}

TEST_F(TestRouter, LambdaRouterConvetable)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::LambdaDynamicRouterComponent<std::string, int, float>>(&even_odd<int>);
    auto sink1  = std::make_shared<node::TestSink<float>>();
    auto sink2  = std::make_shared<node::TestSink<float>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<float>{1.0}), sink1->get_values());
    EXPECT_EQ((std::vector<float>{0.0, 2.0}), sink2->get_values());
}

TEST_F(TestRouter, LambdaRouterConvestion)
{
    auto conversion_call_count = std::atomic<std::size_t>(0);
    auto source                = std::make_shared<node::TestSource<int>>();
    auto router                = std::make_shared<node::LambdaDynamicRouterComponent<std::string, int, std::string>>(
        &even_odd<int>,
        [&conversion_call_count](int&& data) {
            conversion_call_count++;
            return std::to_string(data);
        });
    auto sink1 = std::make_shared<node::TestSink<std::string>>();
    auto sink2 = std::make_shared<node::TestSink<std::string>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ(conversion_call_count, 3);
    EXPECT_EQ((std::vector<std::string>{"1"}), sink1->get_values());
    EXPECT_EQ((std::vector<std::string>{"0", "2"}), sink2->get_values());
}

}  // namespace mrc
