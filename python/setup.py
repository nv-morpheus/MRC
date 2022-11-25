#
# SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages
from setuptools import setup

import versioneer

##############################################################################
# - Python package generation ------------------------------------------------

setup(
    name='mrc',
    description="mrc",
    version=versioneer.get_version(),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    author="NVIDIA Corporation",
    setup_requires=[],
    include_package_data=True,
    packages=find_packages(include=['mrc', 'mrc.*'], exclude=['tests']),
    package_data={
        "mrc": ["_pymrc/*.so"]  # Add the pymrc library for the root package
    },
    license="Apache",
    cmdclass=versioneer.get_cmdclass())
