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

/* TODO commenting out to get it to compile
#include "srf/channel/buffered_channel.hpp"
#include "srf/core/affinity.h"
#include "srf/core/base_node.hpp"

#include <benchmark/benchmark.h>



using namespace srf;

class BenchSink : public SegmentSink<std::uint64_t>
{
    void on_next(std::uint64_t& i) final override { benchmark::DoNotOptimize(++i); }
};

class BenchNode : public GenericNode<std::uint64_t, std::uint64_t>
{
    void on_next(std::uint64_t& i) final override
    {
        benchmark::DoNotOptimize(i++);
        this->await_emit(i);
    }
};

static void SrfBaseSinkChannelWrite(benchmark::State& state)
{
    auto bench_sink = std::make_unique<BenchSink>();
    auto ingress = bench_sink->input_channel();
    auto& ingress_ref = *ingress;
    std::uint64_t counter = 0;
    bench_sink->start();

    auto core_0 = affinity::system::cpu_from_logical_id(0);
    cpu_set cpus;
    cpus.insert(core_0);
    affinity::this_thread::set_affinity(cpus);

    for(auto _ : state)
    {
        ingress_ref.await_write(++counter);
    }

    bench_sink->stop();
    bench_sink->join();
}

static void SrfBaseNodeAndSinkChannelWrite(benchmark::State& state)
{
    auto bench_node = std::make_unique<BenchNode>();
    auto bench_sink = std::make_unique<BenchSink>();

    bench_node->add_edge(bench_sink->input_channel());

    auto ingress = bench_node->input_channel();
    auto& ingress_ref = *ingress;
    std::uint64_t counter = 0;

    bench_sink->start();
    bench_node->start();

    for(auto _ : state)
    {
        ingress_ref.await_write(++counter);
    }

    bench_node->stop();
    bench_sink->stop();

    bench_node->join();
    bench_sink->join();
}

class thread_barrier
{
  private:
    std::size_t initial_;
    std::size_t current_;
    bool cycle_{true};
    std::mutex mtx_{};
    std::condition_variable cond_{};

  public:
    explicit thread_barrier(std::size_t initial) : initial_{initial}, current_{initial_} { BOOST_ASSERT(0 != initial); }

    thread_barrier(thread_barrier const&) = delete;
    thread_barrier& operator=(thread_barrier const&) = delete;

    bool wait()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        const bool cycle = cycle_;
        if(0 == --current_)
        {
            cycle_ = !cycle_;
            current_ = initial_;
            lk.unlock(); // no pessimization
            cond_.notify_all();
            return true;
        }
        cond_.wait(lk, [&]() { return cycle != cycle_; });
        return false;
    }
};

class SrfBaseNodeAndSink : public benchmark::Fixture
{
  public:
    void SetUp(const ::benchmark::State& state)
    {
        bench_node = std::make_unique<BenchNode>();
        bench_sink = std::make_unique<BenchSink>();
        fibers = std::make_shared<FiberPoolTaskQueue>(2, 128);
    }

    void TearDown(const ::benchmark::State& state)
    {
        bench_node->stop();
        bench_sink->stop();

        bench_node->join();
        bench_sink->join();

        fibers.reset();
    }

    std::unique_ptr<BenchNode> bench_node;
    std::unique_ptr<BenchSink> bench_sink;
    std::shared_ptr<FiberPoolTaskQueue> fibers;
};

BENCHMARK_F(SrfBaseNodeAndSink, ChannelWriteMT)(benchmark::State& state)
{
    std::uint64_t counters[state.threads];
    auto& counter = counters[state.thread_index];

    CHECK(bench_node);
    auto ingress = bench_node->input_channel();
    auto& ingress_ref = *ingress;

    if(state.thread_index == 0)
    {
        CHECK(bench_node);
        CHECK(bench_sink);

        bench_node->set_executor(fibers);
        bench_node->concurrency(1);
        bench_sink->set_executor(fibers);
        bench_node->concurrency(1);

        bench_node->add_edge(bench_sink->input_channel());

        bench_sink->start();
        bench_node->start();
    }

    for(auto _ : state)
    {
        ingress_ref.await_write(++counter);
    }

}

BENCHMARK(SrfBaseSinkChannelWrite)->UseRealTime();
BENCHMARK(SrfBaseNodeAndSinkChannelWrite)->UseRealTime();
BENCHMARK_REGISTER_F(SrfBaseNodeAndSink, ChannelWriteMT)->UseRealTime();

 */
