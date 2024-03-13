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

#include "pymrc/utilities/json_values.hpp"

#include "pymrc/utilities/acquire_gil.hpp"
#include "pymrc/utils.hpp"

#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pylifecycle.h>

#include <cstddef>
#include <mutex>
#include <ostream>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace std::literals;

namespace {

void patch_object(py::object& obj,
                  std::vector<std::string>::const_iterator path,
                  std::vector<std::string>::const_iterator path_end,
                  const py::object& value)
{
    if (path == path_end)
    {
        obj = value;
    }
    else if (py::isinstance<py::dict>(obj))
    {
        const auto& key = *path;
        auto py_dict    = obj.cast<py::dict>();
        auto entry      = py_dict[key.c_str()];
        auto next_path  = std::next(path);

        // There are one of two possibilities here:
        // 1. The path is terminal and we should assign value to the dict
        // 2. The path is not terminal and we should recurse into the dict
        if (next_path == path_end)
        {
            entry = value;
        }
        else
        {
            // entry is of type item_accessor, assigning it to a py::object will trigger a __getitem__ call
            py::object next_obj = entry;
            patch_object(next_obj, next_path, path_end, value);
        }
    }
    else if (py::isinstance<py::list>(obj))
    {
        auto py_list     = obj.cast<py::list>();
        const auto index = std::stoul(*path);
        auto next_path   = std::next(path);
        auto entry       = py_list[index];
        if (next_path == path_end)
        {
            entry = value;
        }
        else
        {
            py::object next_obj = entry;
            patch_object(next_obj, next_path, path_end, value);
        }
    }
    else
    {
        throw std::runtime_error("Invalid path");
    }
}
}  // namespace

namespace mrc::pymrc {
JSONValues::JSONValues(py::object values)
{
    AcquireGIL gil;
    m_serialized_values = cast_from_pyobject(values, [this](const py::object& source, const std::string& path) {
        return this->unserializable_handler(source, path);
    });
}

py::object JSONValues::to_python() const
{
    AcquireGIL gil;
    py::object results{cast_from_json(m_serialized_values)};
    for (const auto& [path, obj] : m_py_objects)
    {
        DCHECK(path[0] == '/');
        DVLOG(10) << "Restoring object at path: " << path;
        std::vector<std::string> path_parts;
        boost::split(path_parts, path, boost::is_any_of("/"));

        // Since our paths always begin with a '/', the first element will always be empty in the case where path="/"
        // path_parts will be {"", ""} and we can skip the first element
        auto itr = path_parts.cbegin();
        ++itr;
    }

    return results;
}

nlohmann::json JSONValues::unserializable_handler(const py::object& obj, const std::string& path)
{
    /* We don't know how to serialize the Object, throw it into m_py_objects and return a reference ID*/

    // Remove constness and cache the object
    py::object non_const_copy = obj;
    DVLOG(10) << "Storing unserializable object at path: " << path;
    m_py_objects[path] = std::move(non_const_copy);

    return {"**pymrc_placeholder"s};
}

}  // namespace mrc::pymrc
