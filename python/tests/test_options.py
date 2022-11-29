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

import pytest

import mrc


# @pytest.mark.parametrize("engines_per_pe", [1, 2])
# @pytest.mark.parametrize("pe_count", [1, 3])
@pytest.mark.parametrize(
    "engine_type",
    [mrc.core.options.EngineType.Fiber, mrc.core.options.EngineType.Process, mrc.core.options.EngineType.Thread])
def test_engine_factories_default_engine_type(engine_type: mrc.core.options.EngineType):

    options = mrc.Options()

    # Hold a reference
    eng_factories = options.engine_factories

    eng_factories.default_engine_type = engine_type

    assert eng_factories.default_engine_type == engine_type, "default_engine_type set/get should match on ref"
    assert options.engine_factories.default_engine_type == engine_type, "default_engine_type set/get should match"


def test_engine_factories_dedicated_main_thread():

    options = mrc.Options()

    # Hold a reference
    eng_factories = options.engine_factories

    eng_factories.dedicated_main_thread = True

    assert eng_factories.dedicated_main_thread is True, "dedicated_main_thread set/get should match on ref"
    assert options.engine_factories.dedicated_main_thread is True, "dedicated_main_thread set/get should match"

    eng_factories.dedicated_main_thread = False

    assert eng_factories.dedicated_main_thread is False, "dedicated_main_thread set/get should match on ref"
    assert options.engine_factories.dedicated_main_thread is False, "dedicated_main_thread set/get should match"


@pytest.mark.parametrize(
    "engine_type",
    [mrc.core.options.EngineType.Fiber, mrc.core.options.EngineType.Process, mrc.core.options.EngineType.Thread])
def test_engine_factory_options(engine_type: mrc.core.options.EngineType):

    options = mrc.Options()

    # Hold a reference
    eng_factories = options.engine_factories

    def check_one_option(options: mrc.Options,
                         name: str,
                         cpu_count: int,
                         eng_type: mrc.core.options.EngineType,
                         reusable: bool,
                         allow_overlap: bool):

        option_group = mrc.core.options.EngineFactoryOptions()
        option_group.cpu_count = cpu_count
        option_group.engine_type = eng_type
        option_group.reusable = reusable
        option_group.allow_overlap = allow_overlap

        options.engine_factories.set_engine_factory_options(name, option_group)

        # Get the options
        option_group_ret = eng_factories.engine_group_options(name)

        assert option_group_ret.cpu_count == cpu_count, "cpu_count should match"
        assert option_group_ret.engine_type == eng_type, "engine_type should match"
        assert option_group_ret.reusable == reusable, "reusable should match"
        assert option_group_ret.allow_overlap == allow_overlap, "cpu_count should match"

    # Create single threaded, reusable, allow_overlap
    check_one_option(options, "single_threaded", 1, engine_type, True, True)

    # Create multi threaded, nonreusable, no overlap
    check_one_option(options, "multi_threaded", 2, engine_type, False, False)


if (__name__ == "__main__"):
    test_engine_factories_dedicated_main_thread()
