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

#include <boost/algorithm/string.hpp>  // for split
#include <glog/logging.h>
#include <pybind11/cast.h>

#include <iterator>   // for next
#include <ostream>    // for operator<< needed for logging
#include <sstream>    // for stringstream
#include <stdexcept>  // for runtime_error
#include <utility>    // for move
#include <vector>     // for vector

// We already have <boost/algorithm/string.hpp> included we don't need these others, it is also the only public header
// with a definition for boost::is_any_of, so even if we replaced string.hpp with these others we would still need to
// include string.hpp or a detail/ header
// IWYU pragma: no_include <boost/algorithm/string/classification.hpp>
// IWYU pragma: no_include <boost/algorithm/string/split.hpp>
// IWYU pragma: no_include <boost/iterator/iterator_facade.hpp>

namespace py = pybind11;
using namespace std::string_literals;

namespace {

void patch_object(py::object& obj,
                  std::vector<std::string>::const_iterator path,
                  std::vector<std::string>::const_iterator path_end,
                  const py::object& value)
{
    // Terminal case, assign value to obj
    const auto& path_str = *path;
    if (path_str.empty())
    {
        obj = value;
    }
    else
    {
        // Nested object, since obj is a de-serialized python object the only valid container types will be dict and
        // list. There are one of two possibilities here:
        // 1. The next_path is terminal and we should assign value to the container
        // 2. The next_path is not terminal and we should recurse into the container
        auto next_path = std::next(path);

        if (py::isinstance<py::dict>(obj))
        {
            auto py_dict = obj.cast<py::dict>();
            if (next_path == path_end)
            {
                py_dict[path_str.c_str()] = value;
            }
            else
            {
                py::object next_obj = py_dict[path_str.c_str()];
                patch_object(next_obj, next_path, path_end, value);
            }
        }
        else if (py::isinstance<py::list>(obj))
        {
            auto py_list     = obj.cast<py::list>();
            const auto index = std::stoul(path_str);
            if (next_path == path_end)
            {
                py_list[index] = value;
            }
            else
            {
                py::object next_obj = py_list[index];
                patch_object(next_obj, next_path, path_end, value);
            }
        }
        else
        {
            throw std::runtime_error("Invalid path");
        }
    }
}

void patch_object(py::object& obj, const std::string& path, const py::object& value)
{
    std::vector<std::string> path_parts;
    boost::split(path_parts, path, boost::is_any_of("/"));

    // Since our paths always begin with a '/', the first element will always be empty in the case where path="/"
    // path_parts will be {"", ""} and we can skip the first element
    auto itr = path_parts.cbegin();
    patch_object(obj, std::next(itr), path_parts.cend(), value);
}

void validate_path(const std::string& path)
{
    if (path.empty() || path[0] != '/')
    {
        throw std::runtime_error("Invalid path: " + path);
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

JSONValues::JSONValues(nlohmann::json values) : m_serialized_values(std::move(values)) {}

std::size_t JSONValues::num_unserializable() const
{
    return m_py_objects.size();
}

bool JSONValues::has_unserializable() const
{
    return !m_py_objects.empty();
}

py::object JSONValues::to_python() const
{
    AcquireGIL gil;
    py::object results = cast_from_json(m_serialized_values);
    for (const auto& [path, obj] : m_py_objects)
    {
        DCHECK(path[0] == '/');
        DVLOG(10) << "Restoring object at path: " << path;
        patch_object(results, path, obj);
    }

    return results;
}

nlohmann::json JSONValues::to_json() const
{
    if (const auto num_unserializable = this->num_unserializable(); num_unserializable > 0)
    {
        std::ostringstream error_message;
        error_message << "There are " << num_unserializable << " unserializable objects located at:";
        for (const auto& [path, obj] : m_py_objects)
        {
            error_message << "\n" << path;
        }

        throw std::runtime_error(error_message.str());
    }

    return m_serialized_values;
}

JSONValues JSONValues::set_value(const std::string& path, const pybind11::object& value) const
{
    validate_path(path);

    AcquireGIL gil;
    py::object py_obj = this->to_python();
    patch_object(py_obj, path, value);
    return {py_obj};
}

JSONValues JSONValues::set_value(const std::string& path, nlohmann::json value) const
{
    // Two possibilities:
    // 1) We don't have any unserializable objects, in which case we can just update the JSON object
    // 2) We do have unserializable objects, in which case we need to cast value to python and call the python version
    // of set_value

    if (!has_unserializable())
    {
        validate_path(path);

        // The add operation will update an existing value if it exists, or add a new value if it does not
        // ref: https://datatracker.ietf.org/doc/html/rfc6902#section-4.1
        nlohmann::json patch{{"op", "add"}, {"path", path}, {"value", value}};
        nlohmann::json patches = nlohmann::json::array({std::move(patch)});
        auto new_values        = m_serialized_values.patch(std::move(patches));
        return {std::move(new_values)};
    }

    AcquireGIL gil;
    py::object py_obj = cast_from_json(value);
    return set_value(path, py_obj);
}

nlohmann::json JSONValues::unserializable_handler(const py::object& obj, const std::string& path)
{
    /* We don't know how to serialize the Object, throw it into m_py_objects and return a place-holder */

    // Take a non-const copy of the object
    py::object non_const_copy = obj;
    DVLOG(10) << "Storing unserializable object at path: " << path;
    m_py_objects[path] = std::move(non_const_copy);

    return "**pymrc_placeholder"s;
}

}  // namespace mrc::pymrc
