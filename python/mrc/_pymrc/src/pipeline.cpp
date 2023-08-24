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

#include "pymrc/pipeline.hpp"

#include "pymrc/types.hpp"
#include "pymrc/utils.hpp"

#include "mrc/node/forward.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace mrc::pymrc {

namespace py = pybind11;

namespace {
struct PipelineIngressInfo
{
    std::vector<std::string> m_names;
    std::vector<std::type_index> m_type_indices;
    std::vector<mrc::node::PortUtil::ingress_builder_fn_t> m_ingress_builders;
};

struct PipelineEgressInfo
{
    std::vector<std::string> m_names;
    std::vector<std::type_index> m_type_indices;
    std::vector<mrc::node::PortUtil::egress_builder_fn_t> m_egress_builders;
};

segment::IngressPortsBase collect_ingress_info(py::list ids)
{
    using namespace mrc::node;
    std::vector<std::shared_ptr<segment::IngressPortsBase::port_info_t>> ingress_infos;

    for (const auto& item : ids)
    {
        if (item.get_type().equal(py::str().get_type()))
        {
            VLOG(2) << "Ingress type unspecified, using PyHolder default";

            auto port_util = PortRegistry::find_port_util(typeid(PyHolder));

            ingress_infos.emplace_back(
                std::make_shared<segment::IngressPortsBase::port_info_t>(item.cast<std::string>(),
                                                                         typeid(PyHolder),
                                                                         std::get<0>(port_util->ingress_builders)));
        }
        else if (item.get_type().equal(py::tuple().get_type()))
        {
            auto py_tuple = item.cast<py::tuple>();
            CHECK(py::len(py_tuple) >= 2);
            VLOG(2) << "Ingress type was specified, looking for registered builders.";

            py::str py_name   = py_tuple[0];
            py::type py_type  = py_tuple[1];
            py::bool_ py_bool = (py::len(py_tuple) > 2) ? py_tuple[2] : py::bool_(true);

            bool flag_sp_variant          = py::cast<bool>(py_bool);
            const std::type_info* cpptype = cpptype_info_from_object(py_type);

            bool builder_exists        = (cpptype != nullptr && PortRegistry::has_port_util(*cpptype));
            std::type_index type_index = builder_exists ? *cpptype : typeid(PyHolder);

            auto port_util  = PortRegistry::find_port_util(type_index);
            auto builder_fn = flag_sp_variant ? std::get<1>(port_util->ingress_builders)
                                              : std::get<0>(port_util->ingress_builders);

            ingress_infos.emplace_back(
                std::make_shared<segment::IngressPortsBase::port_info_t>(py_name.cast<std::string>(),
                                                                         type_index,
                                                                         builder_fn));
        }
        else
        {
            throw std::runtime_error("Bad port specification");
        }
    }

    return {std::move(ingress_infos)};
}

segment::EgressPortsBase collect_egress_info(py::list ids)
{
    using namespace mrc::node;
    std::vector<std::shared_ptr<segment::EgressPortsBase::port_info_t>> egress_infos;

    for (const auto& item : ids)
    {
        if (item.get_type().equal(py::str().get_type()))
        {
            VLOG(2) << "Egress type unspecified, using PyHolder default";

            auto port_util = PortRegistry::find_port_util(typeid(PyHolder));

            egress_infos.emplace_back(
                std::make_shared<segment::IngressPortsBase::port_info_t>(item.cast<std::string>(),
                                                                         typeid(PyHolder),
                                                                         std::get<0>(port_util->egress_builders)));
        }
        else if (item.get_type().equal(py::tuple().get_type()))
        {
            auto py_tuple = item.cast<py::tuple>();
            CHECK(py::len(py_tuple) >= 2);
            VLOG(2) << "Egress type was specified, looking for registered builders.";

            // Unpack tuple parameters, must be (name, type, [flag_sp_variant])
            py::str py_name   = py_tuple[0];
            py::type py_type  = py_tuple[1];
            py::bool_ py_bool = (py::len(py_tuple) > 2) ? py_tuple[2] : py::bool_(true);

            bool flag_sp_variant          = py::cast<bool>(py_bool);
            const std::type_info* cpptype = cpptype_info_from_object(py_type);

            bool builder_exists        = (cpptype != nullptr && PortRegistry::has_port_util(*cpptype));
            std::type_index type_index = builder_exists ? *cpptype : typeid(PyHolder);

            auto port_util  = PortRegistry::find_port_util(type_index);
            auto builder_fn = flag_sp_variant ? std::get<1>(port_util->egress_builders)
                                              : std::get<0>(port_util->egress_builders);

            egress_infos.emplace_back(
                std::make_shared<segment::EgressPortsBase::port_info_t>(py_name.cast<std::string>(),
                                                                        type_index,
                                                                        builder_fn));
        }
        else
        {
            throw std::runtime_error("Bad port specification");
        }
    }

    return {std::move(egress_infos)};
}
}  // namespace

Pipeline::Pipeline() : m_pipeline(mrc::make_pipeline()) {}

Pipeline::~Pipeline() = default;

void Pipeline::make_segment(const std::string& name, const std::function<void(mrc::segment::IBuilder&)>& init)
{
    auto init_wrapper = [=](mrc::segment::IBuilder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    m_pipeline->make_segment(name, init_wrapper);
}

void Pipeline::make_segment(const std::string& name,
                            py::list ingress_port_info,
                            py::list egress_port_info,
                            const std::function<void(mrc::segment::IBuilder&)>& init)
{
    if (ingress_port_info.empty() && egress_port_info.empty())
    {
        return make_segment(name, init);
    }

    auto init_wrapper = [init](mrc::segment::IBuilder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    auto ingress_info = collect_ingress_info(ingress_port_info);
    node::PortRegistry::register_name_type_index_pairs(ingress_info.names(), ingress_info.type_indices());

    auto egress_info = collect_egress_info(egress_port_info);
    node::PortRegistry::register_name_type_index_pairs(egress_info.names(), egress_info.type_indices());

    m_pipeline->make_segment(name, std::move(ingress_info), std::move(egress_info), init_wrapper);
}

std::shared_ptr<pipeline::IPipeline> Pipeline::get_wrapped() const
{
    return m_pipeline;
}

}  // namespace mrc::pymrc
