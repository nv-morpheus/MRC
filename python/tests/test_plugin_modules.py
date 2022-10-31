# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import srf

whereami = pathlib.Path(__file__).parent.resolve()
dynamic_module_path = pathlib.Path(f"{whereami}/../../tests/modules").resolve()


def test_plugin_module_import():
    mod = srf.PluginModule.create_or_acquire("doesnt_exist.so")


def test_dynamic_module_plugin_interface():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{dynamic_module_path}")
    plugin_module.load()
    plugin_module.reload()


def test_dynamic_module_plugin_registration():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{dynamic_module_path}")
    plugin_module.load()

    module_namespace = "srf_unittest_cpp_dynamic"
    module_name = "DynamicSourceModule"

    # TODO(bhargav) add tests to ensure that dynamic modules are loaded correctly once registry bindings are in


def test_dynamic_module_bad_version_test():
    pass


if (__name__ in ("__main__",)):
    test_plugin_module_import()
    test_dynamic_module_plugin_interface()
