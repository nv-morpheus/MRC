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

#include "pysrf/port_builders.hpp"
#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>  // for move

namespace srf::pysrf {
namespace py = pybind11;

using ingress_info_t = std::tuple<std::vector<std::string>,
                                  std::vector<std::type_index>,
                                  std::vector<srf::node::PortUtil::ingress_builder_fn_t>>;

using egress_info_t = std::tuple<std::vector<std::string>,
                                 std::vector<std::type_index>,
                                 std::vector<srf::node::PortUtil::egress_builder_fn_t>>;

ingress_info_t collect_ingress_info(py::list ids)
{
    using namespace srf::node;
    std::vector<std::string> port_ids;
    std::vector<std::type_index> port_type_indices;
    std::vector<srf::node::PortUtil::ingress_builder_fn_t> builder_fns;

    for (const auto& item : ids)
    {
        if (item.get_type().equal(py::str().get_type()))
        {
            VLOG(2) << "Ingress type unspecified, using PyHolder default";
            port_ids.push_back(item.cast<std::string>());
            port_type_indices.emplace_back(typeid(PyHolder));

            auto port_util = PortRegistry::find_port_util(typeid(PyHolder));
            builder_fns.push_back(std::get<0>(port_util->m_ingress_builders));
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

            bool builder_exists = (cpptype != nullptr && PortRegistry::has_port_util(*cpptype));
            std::type_index type_index = builder_exists ? *cpptype : typeid(PyHolder);

            auto port_util = PortRegistry::find_port_util(type_index);
            auto builder_fn =
                flag_sp_variant ? std::get<1>(port_util->m_ingress_builders) : std::get<0>(port_util->m_ingress_builders);

            port_ids.push_back(py_name.cast<std::string>());
            port_type_indices.emplace_back(type_index);
            builder_fns.push_back(builder_fn);
        }
        else
        {
            throw std::runtime_error("Bad port specification");
        }
    }

    return {port_ids, port_type_indices, builder_fns};
}

egress_info_t collect_egress_info(py::list ids)
{
    using namespace srf::node;
    std::vector<std::string> port_ids;
    std::vector<std::type_index> port_type_indices;
    std::vector<srf::node::PortUtil::egress_builder_fn_t> builder_fns;

    for (const auto& item : ids)
    {
        if (item.get_type().equal(py::str().get_type()))
        {
            VLOG(2) << "Egress type unspecified, using PyHolder default";
            port_ids.push_back(item.cast<std::string>());
            port_type_indices.emplace_back(typeid(PyHolder));

            auto port_util = PortRegistry::find_port_util(typeid(PyHolder));
            builder_fns.push_back(std::get<0>(port_util->m_egress_builders));
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

            bool builder_exists = (cpptype != nullptr && PortRegistry::has_port_util(*cpptype));
            std::type_index type_index = builder_exists ? *cpptype : typeid(PyHolder);

            auto port_util = PortRegistry::find_port_util(type_index);
            auto builder_fn =
                flag_sp_variant ? std::get<1>(port_util->m_egress_builders) : std::get<0>(port_util->m_egress_builders);

            port_ids.push_back(py_name.cast<std::string>());
            port_type_indices.emplace_back(type_index);
            builder_fns.push_back(builder_fn);
        }
        else
        {
            throw std::runtime_error("Bad port specification");
        }
    }

    return {port_ids, port_type_indices, builder_fns};
}

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
                            py::list ingress_port_info,
                            py::list egress_port_info,
                            const std::function<void(srf::segment::Builder&)>& init)
{
    if (ingress_port_info.empty() && egress_port_info.empty())
    {
        return make_segment(name, init);
    }

    auto init_wrapper = [init](srf::segment::Builder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    auto ingress_info = collect_ingress_info(ingress_port_info);
    segment::IngressPortsBase ingress_ports(std::get<0>(ingress_info), std::get<2>(ingress_info));
    node::PortRegistry::register_name_type_index_pairs(std::get<0>(ingress_info), std::get<1>(ingress_info));

    auto egress_info = collect_egress_info(egress_port_info);
    segment::EgressPortsBase egress_ports(std::get<0>(egress_info), std::get<2>(egress_info));
    node::PortRegistry::register_name_type_index_pairs(std::get<0>(egress_info), std::get<1>(egress_info));

    m_pipeline->make_segment(name, ingress_ports, egress_ports, init_wrapper);
}

std::unique_ptr<srf::pipeline::Pipeline> Pipeline::swap()
{
    auto tmp   = std::move(m_pipeline);
    m_pipeline = srf::pipeline::make_pipeline();
    return std::move(tmp);
}
}  // namespace srf::pysrf
