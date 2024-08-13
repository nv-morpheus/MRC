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

#include "internal/data_plane/data_plane_resources.hpp"

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/fundamental_types.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/when_all.hpp"
#include "mrc/benchmarking/segment_watcher.hpp"
#include "mrc/benchmarking/tracer.hpp"
#include "mrc/benchmarking/util.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/memory/memory_kind.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/memory/resources/host/pinned_memory_resource.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/runtime/remote_descriptor.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>
#include <ucxx/api.h>

#include <any>
#include <coroutine>
#include <vector>

using namespace mrc;
using namespace mrc::benchmarking;
using namespace mrc::memory::literals;

constexpr size_t data_size = 64_KiB;

/**
 * Extends DataPlaneResources2 class to include utility methods
 */
class DataPlaneResources2Tester : public data_plane::DataPlaneResources2
{
  public:
    std::shared_ptr<runtime::Descriptor2> get_descriptor(uint64_t object_id)
    {
        return m_descriptor_by_id.size() ? m_descriptor_by_id[object_id][0] : nullptr;
    }
};

/**
 * Spawns a separate thread to progress UCXX operations
 */
class ProgressEngine
{
public:
    ProgressEngine(DataPlaneResources2Tester& resources): m_resources(resources) {}

    void start_progress()
    {
        m_progress_thread = std::thread([this]() {
            std::unique_lock<std::mutex> lock(m_progress_mutex);
            while (m_is_running)
            {
                m_resources.progress();
                m_progress_cv.wait_for(lock, std::chrono::milliseconds(1)); // Sleep to reduce busy-waiting
            }
        });
    }

    void end_progress()
    {
        m_is_running = false;
        m_progress_cv.notify_all(); // Wake up the progress thread to exit
        m_progress_thread.join();
    }

private:
    DataPlaneResources2Tester& m_resources;

    std::atomic<bool> m_is_running = true;
    std::mutex m_progress_mutex;
    std::condition_variable m_progress_cv;
    std::thread m_progress_thread;
};

/**
 * Serialization and deserialization methods for vector objects allocated on Host or Device memory.
 * Note that although std::vector is used for Host memory and unsigned char* is used for Device memory,
 * the slight overhead of cudaMalloc and cudaMemcpy for Device memory does not significantly affect benchmark results.
 */
namespace mrc::codable {
template <typename T>
struct codable_protocol<std::vector<T>>
{
    static void serialize(const std::vector<T>& obj,
                          mrc::codable::Encoder2<std::vector<T>>& encoder)
    {
        // First put in the size
        mrc::codable::encode2(obj.size(), encoder);

        if constexpr (std::is_fundamental_v<T>)
        {
            // Since these are fundamental types, just encode in a single memory block
            encoder.write_descriptor({obj.data(), obj.size() * sizeof(T), memory::memory_kind::host});
        }
        else
        {
            // Now encode each object
            for (const auto& o : obj)
            {
                mrc::codable::encode2(o, encoder);
            }
        }
    }

    static std::vector<T> deserialize(const Decoder2<std::vector<T>>& decoder)
    {
        auto count = mrc::codable::decode2<size_t>(decoder);

        auto object = std::vector<T>(count);

        decoder.read_descriptor({object.data(), count * sizeof(T), memory::memory_kind::host});

        return object;
    }
};

template <>
struct codable_protocol<unsigned char*>
{
    static void serialize(const unsigned char* obj,
                          mrc::codable::Encoder2<unsigned char*>& encoder)
    {
        // First put in the size
        mrc::codable::encode2(data_size, encoder);

        // Since these are fundamental types, just encode in a single memory block
        encoder.write_descriptor({obj, data_size * sizeof(unsigned char), memory::memory_kind::device});
    }

    static unsigned char* deserialize(const Decoder2<unsigned char*>& decoder)
    {
        size_t size = mrc::codable::decode2<size_t>(decoder);

        unsigned char* object;
        cudaMalloc((void**)&object, size * sizeof(unsigned char));

        decoder.read_descriptor({object, size * sizeof(unsigned char), memory::memory_kind::device});

        return object;
    }
};
}  // namespace mrc::codable

struct IObject
{
    virtual ~IObject() = default;
    virtual std::any get_object() = 0;
    virtual void register_buffer(std::any recv_data) = 0;
};

class HostObject final : public IObject
{
  public:
    HostObject(std::vector<u_int8_t> obj): m_obj(std::move(obj)) {}

    std::any get_object() override
    {
        return std::ref(m_obj);
    }

    void register_buffer(std::any recv_data) override {}

  private:
    std::vector<u_int8_t> m_obj;
};

class DeviceObject final : public IObject
{
  public:
    DeviceObject(std::vector<u_int8_t> obj)
    {
        cudaMalloc(&m_obj, obj.size() * sizeof(u_int8_t));
        cudaMemcpy(m_obj, obj.data(), obj.size() * sizeof(u_int8_t), cudaMemcpyHostToDevice);
    }

    ~DeviceObject()
    {
        cudaFree(m_obj);
        cudaFree(m_res);
    }

    std::any get_object() override
    {
        return std::ref(m_obj);
    }

    void register_buffer(std::any recv_data) override
    {
        m_res = std::any_cast<u_int8_t*>(recv_data);
    }

  private:
    u_int8_t* m_obj;

    // Store the result buffer to free cuda allocated memory during teardown
    u_int8_t* m_res;
};

class DescriptorFixture : public benchmark::Fixture
{
  public:
    void SetUp(const benchmark::State& state) override
    {
        cudaSetDevice(0); // Ensure the CUDA context is initialized

        if (!m_resources)
        {
            m_resources = std::make_unique<DataPlaneResources2Tester>();

            m_resources->set_instance_id(42);

            m_loopback_endpoint = m_resources->create_endpoint(m_resources->address(), m_resources->get_instance_id());
        }

        m_progress_engine = std::unique_ptr<ProgressEngine>(new ProgressEngine(*m_resources));

        // Setup host object and device object using IObject interface
        m_kind = memory::memory_kind(state.range(0));
        for (int i = 0; i < state.range(1); ++i)
        {
            std::vector<u_int8_t> send_data(data_size);
            switch (m_kind)
            {
                case memory::memory_kind::host:
                    m_obj.emplace_back(std::unique_ptr<IObject>(new HostObject(send_data)));
                    break;

                case memory::memory_kind::device:
                    m_obj.emplace_back(std::unique_ptr<IObject>(new DeviceObject(send_data)));
                    break;
            }
        }
    }

    void TearDown(const benchmark::State& state) override
    {
        m_obj.clear();
        m_progress_engine.reset();
    }

    template <typename T>
    void TransferFullDescriptors(size_t messages_to_send)
    {
        m_progress_engine->start_progress();

        auto send_thread = std::thread([&]() {
            cudaSetDevice(0); // Ensure the CUDA context is initialized

            // Store send_requests to check for completion after all messages are sent to achieve async sending
            std::vector<coroutines::Task<std::shared_ptr<ucxx::Request>>> pending_tasks;

            std::vector<std::weak_ptr<runtime::Descriptor2>> registered_send_descriptors;
            for (size_t i = 0; i < messages_to_send; ++i)
            {
                auto send_data = std::any_cast<std::reference_wrapper<T>>(m_obj[i]->get_object()).get();
                auto send_descriptor = runtime::Descriptor2::create_from_value(std::move(send_data), *m_resources);

                auto send_request = m_resources->await_send_descriptor(send_descriptor, m_loopback_endpoint);
                pending_tasks.push_back(std::move(send_request));

                auto send_descriptor_object_id = send_descriptor->encoded_object().object_id();

                send_descriptor = nullptr;

                // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
                // invalid once `DataPlaneResources2` releases it
                registered_send_descriptors.push_back(m_resources->get_descriptor(send_descriptor_object_id));
            }

            // Wait for all send requests to complete before proceeding
            coroutines::sync_wait(coroutines::when_all(std::move(pending_tasks)));

            // Wait for remote decrement messages
            // This loop also guarantees that all send_requests and recv_requests have been completed
            for (auto& registered_send_descriptor : registered_send_descriptors)
            {
                while (registered_send_descriptor.lock() != nullptr)
                {
                    std::this_thread::yield(); // Yield to avoid busy-waiting
                }
            }
        });

        // Receive the messages, process them into descriptors, and deserialize into class T objects
        for (size_t i = 0; i < messages_to_send; i++) {
            std::shared_ptr<runtime::Descriptor2> recv_descriptor = coroutines::sync_wait(m_resources->await_recv_descriptor());

            auto recv_data = recv_descriptor->deserialize<T>();

            m_obj[i]->register_buffer(recv_data);
        }

        send_thread.join();

        m_progress_engine->end_progress();
    }

    void run_benchmark(size_t messages_to_send)
    {
        (m_kind == memory::memory_kind::device) ? TransferFullDescriptors<u_int8_t*>(messages_to_send)
                                                : TransferFullDescriptors<std::vector<u_int8_t>>(messages_to_send);
    }

  private:
    std::unique_ptr<DataPlaneResources2Tester> m_resources;
    std::shared_ptr<ucxx::Endpoint> m_loopback_endpoint;

    std::unique_ptr<ProgressEngine> m_progress_engine;

    std::vector<std::unique_ptr<IObject>> m_obj;
    memory::memory_kind m_kind;
};

BENCHMARK_DEFINE_F(DescriptorFixture, descriptor_latency)(benchmark::State& state)
{
    size_t messages_to_send = state.range(1);
    std::chrono::duration<double> total_time{0};

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        this->run_benchmark(messages_to_send);

        auto end = std::chrono::high_resolution_clock::now();
        total_time += (end - start);
    }

    state.counters["average_latency"] = total_time.count() / messages_to_send;
    state.counters["bytes_per_second"] = messages_to_send * data_size * state.iterations() / total_time.count();
    state.counters["messages_per_second"] = messages_to_send * state.iterations() / total_time.count();
}

/*
 * NOTE: SetUp and TearDown logic must be repeated after each successive iteration, increasing difficulty and accurately
 * measuring latency and throughput. A single iteration with x messages sent per experiment should suffice.
 */

// Benchmark device memory with range of messages_to_send
BENCHMARK_REGISTER_F(DescriptorFixture, descriptor_latency)
    ->ArgsProduct({
        {static_cast<int>(memory::memory_kind::device)}, // memory type
        benchmark::CreateRange(1, 10000, /*multi=*/10),   // number of messages to send
    })
    ->Iterations(1);

// Benchmark host memory with range of messages_to_send
BENCHMARK_REGISTER_F(DescriptorFixture, descriptor_latency)
    ->ArgsProduct({
        {static_cast<int>(memory::memory_kind::host)}, // memory type
        benchmark::CreateRange(1, 10000, /*multi=*/10),   // number of messages to send
    })
    ->Iterations(1);
