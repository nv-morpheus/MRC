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

#include "pysrf/pipeline.hpp"

#include "pysrf/types.hpp"

#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/egress_ports.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
namespace py = pybind11;
using namespace srf::pysrf;

template <std::size_t Count, template <class...> class PortClass, typename... ArgsT>
struct PipelinePortBuilder : PipelinePortBuilder<Count - 1, PortClass, PyHolder, ArgsT...>
{};

template <template <class...> class PortClass, typename... ArgsT>
struct PipelinePortBuilder<0, PortClass, ArgsT...>
{
    using port_type_t                 = PortClass<ArgsT...>;
    static constexpr size_t PortCount = sizeof...(ArgsT);

    static port_type_t build(const std::vector<std::string>& port_ids)
    {
        return port_type_t(port_ids);
    }
};
}  // namespace

namespace srf::pysrf {

namespace py = pybind11;

Pipeline::Pipeline() : m_pipeline(srf::pipeline::make_pipeline()) {}

void Pipeline::make_segment(const std::string& name, const std::function<void(srf::segment::Builder&)>& init)
{
    auto init_wrapper = [=](srf::segment::Builder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    m_pipeline->make_segment(name, init_wrapper);
}

void Pipeline::make_segment(const std::string& name,
                            const std::vector<std::string>& ingress_port_ids,
                            const std::vector<std::string>& egress_port_ids,
                            const std::function<void(srf::segment::Builder&)>& init)
{
    if (ingress_port_ids.empty() && egress_port_ids.empty())
    {
        return make_segment(name, init);
    }

    auto init_wrapper = [init](srf::segment::Builder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    if (!ingress_port_ids.empty() && !egress_port_ids.empty())
    {
        dynamic_port_config(name, ingress_port_ids, egress_port_ids, init_wrapper);
    }
    else if (!ingress_port_ids.empty())
    {
        dynamic_port_config_ingress(name, ingress_port_ids, init);
    }
    else
    {
        dynamic_port_config_egress(name, egress_port_ids, init);
    }
}

// Need to have all supported cases explicitly enumerated so python bindings work as expected.
void Pipeline::dynamic_port_config(const std::string& name,
                                   const std::vector<std::string>& ingress_port_ids,
                                   const std::vector<std::string>& egress_port_ids,
                                   const std::function<void(srf::segment::Builder&)>& init)
{
    if (ingress_port_ids.empty() || ingress_port_ids.size() > SRF_MAX_INGRESS_PORTS)
    {
        throw std::runtime_error("Maximum of 10 egress ports supported via python interface");
    }

    if (ingress_port_ids.size() == 1)
    {
        auto ingress_ports = ::PipelinePortBuilder<1, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 2)
    {
        auto ingress_ports = ::PipelinePortBuilder<2, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 3)
    {
        auto ingress_ports = ::PipelinePortBuilder<3, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 4)
    {
        auto ingress_ports = ::PipelinePortBuilder<4, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 5)
    {
        auto ingress_ports = ::PipelinePortBuilder<5, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 6)
    {
        auto ingress_ports = ::PipelinePortBuilder<6, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 7)
    {
        auto ingress_ports = ::PipelinePortBuilder<7, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 8)
    {
        auto ingress_ports = ::PipelinePortBuilder<8, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 9)
    {
        auto ingress_ports = ::PipelinePortBuilder<9, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
    else if (ingress_port_ids.size() == 10)
    {
        auto ingress_ports = ::PipelinePortBuilder<10, segment::IngressPorts>::build(ingress_port_ids);
        typed_dynamic_port_config_egress(name, ingress_ports, egress_port_ids, init);
    }
}

// Need to have all supported cases enumerated for python
void Pipeline::dynamic_port_config_ingress(const std::string& name,
                                           const std::vector<std::string>& ingress_port_ids,
                                           const std::function<void(srf::segment::Builder&)>& init)
{
    if (ingress_port_ids.empty() || ingress_port_ids.size() > SRF_MAX_INGRESS_PORTS)
    {
        throw std::runtime_error("Maximum of 10 egress ports supported via python interface");
    }

    if (ingress_port_ids.size() == 1)
    {
        auto ingress_ports = ::PipelinePortBuilder<1, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 2)
    {
        auto ingress_ports = ::PipelinePortBuilder<2, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 3)
    {
        auto ingress_ports = ::PipelinePortBuilder<3, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 4)
    {
        auto ingress_ports = ::PipelinePortBuilder<4, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 5)
    {
        auto ingress_ports = ::PipelinePortBuilder<5, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 6)
    {
        auto ingress_ports = ::PipelinePortBuilder<6, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 7)
    {
        auto ingress_ports = ::PipelinePortBuilder<7, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 8)
    {
        auto ingress_ports = ::PipelinePortBuilder<8, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 9)
    {
        auto ingress_ports = ::PipelinePortBuilder<9, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else if (ingress_port_ids.size() == 10)
    {
        auto ingress_ports = ::PipelinePortBuilder<10, segment::IngressPorts>::build(ingress_port_ids);
        m_pipeline->make_segment(name, ingress_ports, init);
    }
    else
    {
        throw std::runtime_error("Maximum of 10 ingress ports supported via python interface");
    }
}

// Need to have all supported cases enumerated for python
void Pipeline::dynamic_port_config_egress(const std::string& name,
                                          const std::vector<std::string>& egress_port_ids,
                                          const std::function<void(srf::segment::Builder&)>& init)
{
    if (egress_port_ids.empty() || egress_port_ids.size() > SRF_MAX_EGRESS_PORTS)
    {
        throw std::runtime_error("Maximum of 10 egress ports supported via python interface");
    }

    if (egress_port_ids.size() == 1)
    {
        auto egress_ports = ::PipelinePortBuilder<1, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 2)
    {
        auto egress_ports = ::PipelinePortBuilder<2, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 3)
    {
        auto egress_ports = ::PipelinePortBuilder<3, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 4)
    {
        auto egress_ports = ::PipelinePortBuilder<4, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 5)
    {
        auto egress_ports = ::PipelinePortBuilder<5, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 6)
    {
        auto egress_ports = ::PipelinePortBuilder<6, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 7)
    {
        auto egress_ports = ::PipelinePortBuilder<7, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 8)
    {
        auto egress_ports = ::PipelinePortBuilder<8, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 9)
    {
        auto egress_ports = ::PipelinePortBuilder<9, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
    else if (egress_port_ids.size() == 10)
    {
        auto egress_ports = ::PipelinePortBuilder<10, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->make_segment(name, egress_ports, init);
    }
}

template <typename... IngressPortTypesT>
void Pipeline::typed_dynamic_port_config_egress(const std::string& name,
                                                const segment::IngressPorts<IngressPortTypesT...>& ingress_ports,
                                                const std::vector<std::string>& egress_port_ids,
                                                const std::function<void(srf::segment::Builder&)>& init)
{
    if (egress_port_ids.empty() || egress_port_ids.size() > SRF_MAX_EGRESS_PORTS)
    {
        throw std::runtime_error("Maximum of 10 egress ports supported via python interface");
    }

    if (egress_port_ids.size() == 1)
    {
        auto egress_ports = ::PipelinePortBuilder<1, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 2)
    {
        auto egress_ports = ::PipelinePortBuilder<2, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 3)
    {
        auto egress_ports = ::PipelinePortBuilder<3, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 4)
    {
        auto egress_ports = ::PipelinePortBuilder<4, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 5)
    {
        auto egress_ports = ::PipelinePortBuilder<5, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 6)
    {
        auto egress_ports = ::PipelinePortBuilder<6, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 7)
    {
        auto egress_ports = ::PipelinePortBuilder<7, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 8)
    {
        auto egress_ports = ::PipelinePortBuilder<8, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 9)
    {
        auto egress_ports = ::PipelinePortBuilder<9, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
    else if (egress_port_ids.size() == 10)
    {
        auto egress_ports = ::PipelinePortBuilder<10, segment::EgressPorts>::build(egress_port_ids);
        m_pipeline->template make_segment(name, ingress_ports, egress_ports, init);
    }
}

std::unique_ptr<srf::pipeline::Pipeline> Pipeline::swap()
{
    auto tmp   = std::move(m_pipeline);
    m_pipeline = srf::pipeline::make_pipeline();
    return std::move(tmp);
}
}  // namespace srf::pysrf
