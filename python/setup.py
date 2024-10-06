#
# SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from setuptools import find_namespace_packages
from setuptools import setup

import versioneer

##############################################################################
# - Python package generation ------------------------------------------------

setup(name="mrc",
      description="mrc",
      version=versioneer.get_version(),
      classifiers=[
          "Intended Audience :: Developers", "Programming Language :: Python", "Programming Language :: Python :: 3.10"
      ],
      author="NVIDIA Corporation",
      setup_requires=[],
      include_package_data=True,
      packages=find_namespace_packages(include=["mrc*"], exclude=["tests", "mrc.core.segment.module_definitions"]),
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False)
