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

#include "internal/pubsub/publisher_manager.hpp"

namespace srf::internal::pubsub {

PublisherBackend::PublisherBackend(std::string name, runtime::Runtime& runtime) : PubSubBase(std::move(name), runtime)
{}

PublisherBackend::~PublisherBackend()
{
    service_await_join();
}
const std::string& PublisherBackend::role() const
{
    return role_publisher();
}
const std::set<std::string>& PublisherBackend::subscribe_to_roles() const
{
    static std::set<std::string> r = {role_subscriber()};
    return r;
}
const std::unordered_map<std::uint64_t, InstanceID>& PublisherBackend::tagged_instances() const
{
    return m_tagged_instances;
}
const std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>& PublisherBackend::tagged_endpoints() const
{
    return m_tagged_endpoints;
}
void PublisherBackend::do_service_start()
{
    SubscriptionService::do_service_start();

    CHECK(this->tag() != 0);

    auto drop_subscription_service_lambda = drop_subscription_service();

    auto publisher = std::unique_ptr<Publisher>(new Publisher(service_name(), this->tag(), resources()));

    auto sink = std::make_unique<srf::node::RxSink<element_type>>(
        [this](element_type encoded_object) { write(std::move(encoded_object)); });

    srf::node::make_edge(*publisher, *sink);

    auto launch_options = resources().network()->control_plane().client().launch_options();
    m_writer = resources().runnable().launch_control().prepare_launcher(launch_options, std::move(sink))->ignition();

    m_publisher_promise.set_value(std::move(publisher));

    SRF_THROW_ON_ERROR(activate_subscription_service());
}
void PublisherBackend::do_service_await_live()
{
    m_writer->await_live();
}
void PublisherBackend::do_service_stop()
{
    m_writer->stop();
}
void PublisherBackend::do_service_kill()
{
    m_writer->kill();
}
void PublisherBackend::do_service_await_join()
{
    m_writer->await_join();
}
}  // namespace srf::internal::pubsub
