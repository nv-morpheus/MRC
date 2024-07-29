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
#include <vector>

using namespace mrc;
using namespace mrc::benchmarking;
using namespace mrc::memory::literals;

constexpr size_t data_size = 64_KiB;

class DataPlaneResources2Tester : public data_plane::DataPlaneResources2
{
  public:
    std::shared_ptr<runtime::Descriptor2> get_descriptor(uint64_t object_id)
    {
        return m_descriptor_by_id.size() ? m_descriptor_by_id[object_id][0] : nullptr;
    }
};

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

// Serialization methods meant for testing device allocated memory
template <>
struct codable_protocol<unsigned char*>
{
    static void serialize(const unsigned char* obj,
                          mrc::codable::Encoder2<unsigned char*>& encoder)
    {
        mrc::codable::encode2(data_size, encoder);

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
        m_resources = std::make_unique<DataPlaneResources2Tester>();

        m_resources->set_instance_id(42);

        m_loopback_endpoint = m_resources->create_endpoint(m_resources->address(), m_resources->get_instance_id());

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
        m_resources.reset();
    }

    template <typename T>
    void TransferFullDescriptors(benchmark::State& state)
    {
        std::atomic<bool> is_running(true);
        std::mutex progress_mutex;
        std::condition_variable progress_cv;

        auto progress_thread = std::thread([&]() {
            std::unique_lock<std::mutex> lock(progress_mutex);
            while (is_running)
            {
                m_resources->progress();
                progress_cv.wait_for(lock, std::chrono::milliseconds(1)); // Sleep to reduce busy-waiting
            }
        });

        size_t messages_to_send = state.range(1);
        auto receive_thread = std::thread([&]() {
            for (size_t i = 0; i < messages_to_send; i++)
            {
                auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);

                while (!receive_request->isCompleted())
                {
                    std::this_thread::yield(); // Yield to avoid busy-waiting
                }

                auto buffer_view = memory::buffer_view(receive_request->getRecvBuffer()->data(),
                                                       receive_request->getRecvBuffer()->getSize(),
                                                       mrc::memory::memory_kind::host);

                std::shared_ptr<runtime::Descriptor2> recv_descriptor = runtime::Descriptor2::create(buffer_view,
                                                                                                     *m_resources);

                auto recv_data = recv_descriptor->deserialize<T>();

                m_obj[i]->register_buffer(recv_data);
            }
        });

        std::vector<std::future<void>> send_futures;
        std::vector<std::weak_ptr<runtime::Descriptor2>> registered_send_descriptors;


        auto send_function = [&](size_t i) {
            cudaSetDevice(0); // Ensure the CUDA context is initialized

            auto send_data = std::any_cast<std::reference_wrapper<T>>(m_obj[i]->get_object()).get();
            std::shared_ptr<runtime::Descriptor2> send_descriptor = runtime::Descriptor2::create(std::move(send_data),
                                                                                                 *m_resources);

            // Get the serialized data
            auto serialized_data           = send_descriptor->serialize<T>(memory::malloc_memory_resource::instance());
            auto send_descriptor_object_id = send_descriptor->encoded_object().object_id();

            send_descriptor = nullptr;

            auto send_request = m_resources->am_send_async(m_loopback_endpoint, serialized_data);

            // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
            // invalid once `DataPlaneResources2` releases it.
            registered_send_descriptors.push_back(m_resources->get_descriptor(send_descriptor_object_id));

            while (!send_request->isCompleted())
            {
                std::this_thread::yield(); // Yield to avoid busy-waiting
            }
        };

        for (size_t i = 0; i < messages_to_send; ++i)
        {
            send_futures.push_back(std::async(std::launch::async, send_function, i));
        }

        // Wait for all send operations to complete
        for (auto& future : send_futures)
        {
            future.get();
        }

        // Wait for remote decrement messages.
        for (auto& registered_send_descriptor : registered_send_descriptors)
        {
            while (registered_send_descriptor.lock() != nullptr)
            {
                std::this_thread::yield(); // Yield to avoid busy-waiting
            }
        }

        is_running = false;
        progress_cv.notify_all(); // Wake up the progress thread to exit

        receive_thread.join();
        progress_thread.join();
    }

    memory::memory_kind m_kind;

  private:
    std::unique_ptr<DataPlaneResources2Tester> m_resources;
    std::shared_ptr<ucxx::Endpoint> m_loopback_endpoint;

    std::vector<std::unique_ptr<IObject>> m_obj;
};

BENCHMARK_DEFINE_F(DescriptorFixture, descriptor_latency)(benchmark::State& state)
{
    size_t messages_to_send = state.range(1);

    for (auto _ : state)
    {
        (m_kind == memory::memory_kind::device) ? TransferFullDescriptors<u_int8_t*>(state)
                                                : TransferFullDescriptors<std::vector<u_int8_t>>(state);
    }

    state.SetBytesProcessed(messages_to_send * data_size * state.iterations());
    state.SetItemsProcessed(messages_to_send * state.iterations());
}

// SetUp and TearDown logic must be repeated after each successive iteration, increasing difficulty and accurately
// measuring latency and throughput. A single iteration with x messages sent per experiment should suffice.
BENCHMARK_REGISTER_F(DescriptorFixture, descriptor_latency)
    ->ArgsProduct({
        {static_cast<int>(memory::memory_kind::host),
         static_cast<int>(memory::memory_kind::device)}, // memory type
        benchmark::CreateRange(1, 1000, /*multi=*/10),   // number of messages to send
    })
    ->Iterations(1);
