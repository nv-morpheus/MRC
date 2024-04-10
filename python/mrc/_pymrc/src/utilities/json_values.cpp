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

#include "mrc/utils/string_utils.hpp"  // for MRC_CONCAT_STR, split_string_to_array

#include <glog/logging.h>
#include <pybind11/cast.h>

#include <functional>  // for function
#include <iterator>    // for next
#include <map>         // for map
#include <sstream>     // for operator<< & stringstream
#include <stdexcept>   // for runtime_error
#include <utility>     // for move
#include <vector>      // for vector

namespace py = pybind11;
using namespace std::string_literals;

namespace {

std::vector<std::string> split_path(const std::string& path)
{
    return mrc::split_string_to_vector(path, "/"s);
}

struct PyFoundObject
{
    py::object obj;
    py::object index = py::none();
};

PyFoundObject find_object_at_path(py::object& obj,
                                  std::vector<std::string>::const_iterator path,
                                  std::vector<std::string>::const_iterator path_end)
{
    // Terminal case
    const auto& path_str = *path;
    if (path_str.empty())
    {
        return PyFoundObject(obj);
    }

    // Nested object, since obj is a de-serialized python object the only valid container types will be dict and
    // list. There are one of two possibilities here:
    // 1. The next_path is terminal and we should assign value to the container
    // 2. The next_path is not terminal and we should recurse into the container
    auto next_path = std::next(path);

    if (py::isinstance<py::dict>(obj) || py::isinstance<py::list>(obj))
    {
        py::object index;
        if (py::isinstance<py::dict>(obj))
        {
            index = py::cast(path_str);
        }
        else
        {
            index = py::cast(std::stoul(path_str));
        }

        if (next_path == path_end)
        {
            return PyFoundObject{obj, index};
        }

        py::object next_obj = obj[index];
        return find_object_at_path(next_obj, next_path, path_end);
    }

    throw std::runtime_error("Invalid path");
}

PyFoundObject find_object_at_path(py::object& obj, const std::string& path)
{
    auto path_parts = split_path(path);

    // Since our paths always begin with a '/', the first element will always be empty in the case where path="/"
    // path_parts will be {"", ""} and we can skip the first element
    auto itr = path_parts.cbegin();
    return find_object_at_path(obj, std::next(itr), path_parts.cend());
}

void patch_object(py::object& obj, const std::string& path, const py::object& value)
{
    if (path == "/")
    {
        // Special case for the root object since find_object_at_path will return a copy not a reference we need to
        // perform the assignment here
        obj = value;
    }
    else
    {
        auto found = find_object_at_path(obj, path);
        DCHECK(!found.index.is_none());
        found.obj[found.index] = value;
    }
}

std::string validate_path(const std::string& path)
{
    if (path.empty() || path[0] != '/')
    {
        return "/" + path;
    }

    return path;
}
}  // namespace

namespace mrc::pymrc {
JSONValues::JSONValues() : JSONValues(nlohmann::json()) {}

JSONValues::JSONValues(py::object values)
{
    AcquireGIL gil;
    m_serialized_values = cast_from_pyobject(values, [this](const py::object& source, const std::string& path) {
        return this->unserializable_handler(source, path);
    });
}

JSONValues::JSONValues(nlohmann::json values) : m_serialized_values(std::move(values)) {}

JSONValues::JSONValues(nlohmann::json&& values, python_map_t&& py_objects) :
  m_serialized_values(std::move(values)),
  m_py_objects(std::move(py_objects))
{}

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

nlohmann::json::const_reference JSONValues::view_json() const
{
    return m_serialized_values;
}

nlohmann::json JSONValues::to_json(unserializable_handler_fn_t unserializable_handler_fn) const
{
    // start with a copy
    nlohmann::json json_doc = m_serialized_values;
    nlohmann::json patches  = nlohmann::json::array();
    for (const auto& [path, obj] : m_py_objects)
    {
        nlohmann::json patch{{"op", "replace"}, {"path", path}, {"value", unserializable_handler_fn(obj, path)}};
        patches.emplace_back(std::move(patch));
    }

    if (!patches.empty())
    {
        json_doc.patch_inplace(patches);
    }

    return json_doc;
}

JSONValues JSONValues::operator[](const std::string& path) const
{
    auto validated_path = validate_path(path);

    if (validated_path == "/")
    {
        return *this;  // Return a copy of the object
    }

    nlohmann::json::json_pointer node_json_ptr(validated_path);
    if (!m_serialized_values.contains(node_json_ptr))
    {
        throw std::runtime_error(MRC_CONCAT_STR("Path: '" << path << "' not found in json"));
    }

    // take a copy of the sub-object
    nlohmann::json value = m_serialized_values[node_json_ptr];
    python_map_t py_objects;
    for (const auto& [py_path, obj] : m_py_objects)
    {
        if (py_path.find(validated_path) == 0)
        {
            py_objects[py_path] = obj;
        }
    }

    return {std::move(value), std::move(py_objects)};
}

pybind11::object JSONValues::get_python(const std::string& path) const
{
    return (*this)[path].to_python();
}

nlohmann::json JSONValues::get_json(const std::string& path,
                                    unserializable_handler_fn_t unserializable_handler_fn) const
{
    return (*this)[path].to_json(unserializable_handler_fn);
}

nlohmann::json JSONValues::stringify(const pybind11::object& obj, const std::string& path)
{
    AcquireGIL gil;
    return py::str(obj).cast<std::string>();
}

JSONValues JSONValues::set_value(const std::string& path, const pybind11::object& value) const
{
    AcquireGIL gil;
    py::object py_obj = this->to_python();
    patch_object(py_obj, validate_path(path), value);
    return {py_obj};
}

JSONValues JSONValues::set_value(const std::string& path, nlohmann::json value) const
{
    // Two possibilities:
    // 1) We don't have any unserializable objects, in which case we can just update the JSON object
    // 2) We do have unserializable objects, in which case we need to cast value to python and call the python
    // version of set_value

    if (!has_unserializable())
    {
        // The add operation will update an existing value if it exists, or add a new value if it does not
        // ref: https://datatracker.ietf.org/doc/html/rfc6902#section-4.1
        nlohmann::json patch{{"op", "add"}, {"path", validate_path(path)}, {"value", value}};
        nlohmann::json patches = nlohmann::json::array({std::move(patch)});
        auto new_values        = m_serialized_values.patch(std::move(patches));
        return {std::move(new_values)};
    }

    AcquireGIL gil;
    py::object py_obj = cast_from_json(value);
    return set_value(path, py_obj);
}

JSONValues JSONValues::set_value(const std::string& path, const JSONValues& value) const
{
    if (value.has_unserializable())
    {
        AcquireGIL gil;
        py::object py_obj = value.to_python();
        return set_value(path, py_obj);
    }

    return set_value(path, value.to_json([](const py::object& source, const std::string& path) {
        DLOG(FATAL) << "Should never be called";
        return nlohmann::json();  // unreachable but needed to satisfy the signature
    }));
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
