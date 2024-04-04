# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
from decimal import Decimal

import pytest

from mrc.tests.utils import roundtrip_cast


def test_docstrings():
    expected_docstring = "roundtrip_cast(v: object) -> object"
    docstring = inspect.getdoc(roundtrip_cast)
    assert docstring == expected_docstring


@pytest.mark.parametrize(
    "value",
    [
        12,
        2.4,
        RuntimeError("test"),
        Decimal("1.2"),
        "test", [1, 2, 3], {
            "a": 1, "b": 2
        }, {
            "a": 1, "b": RuntimeError("not serializable")
        }, {
            "a": 1, "b": Decimal("1.3")
        }
    ],
    ids=["int", "float", "exception", "decimal", "str", "list", "dict", "dict_w_exception", "dict_w_decimal"])
def test_cast_roundtrip(value: object):
    result = roundtrip_cast(value)
    assert result == value
