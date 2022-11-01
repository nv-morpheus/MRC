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

import logging

import pytest
import srf


def test_module_registry_contains():
    registry = srf.ModuleRegistry()

    print(f"Module registry contains 'xyz': {registry.contains_namespace('xyz')}")


def module_init_fn(builder: srf.Builder):
    print("Called module_init_fn")
    pass


def test_module_registry_register_bad_version():
    registry = srf.ModuleRegistry()

    with pytest.raises(Exception):
        registry.register_module("a_module", "srf_unittests", [22, 19, 0], module_init_fn)


def test_module_registry_register_good_version():
    registry = srf.ModuleRegistry()

    registry.register_module("a_module", "srf_unittests", [22, 11, 0], module_init_fn)


if (__name__ in ("__main__",)):
    test_module_registry_contains()
    test_module_registry_register_bad_version()
    test_module_registry_register_good_version()
