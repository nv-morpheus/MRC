#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh
/usr/bin/nvidia-smi

update_conda_env

rapids-logger "Fetching Build artifacts from ${DISPLAY_ARTIFACT_URL}/"
download_artifact "dot_cache-${REAL_ARCH}.tar.bz"
download_artifact "build-${REAL_ARCH}.tar.bz"

tar xf "${WORKSPACE_TMP}/dot_cache-${REAL_ARCH}.tar.bz"
tar xf "${WORKSPACE_TMP}/build-${REAL_ARCH}.tar.bz"

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

rapids-logger "Installing MRC"
cmake -P ${MRC_ROOT}/build/cmake_install.cmake
pip install ${MRC_ROOT}/build/python

git submodule update --init --recursive
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} .


rapids-logger "Running C++ Tests"
cd ${MRC_ROOT}/build
set +e
ctest --output-on-failure \
      --output-junit ${REPORTS_DIR}/report_ctest.xml

CTEST_RESULTS=$?
set -e

rapids-logger "Running Python Tests"
cd ${MRC_ROOT}/build/python
set +e
pytest -v --junit-xml=${WORKSPACE_TMP}/report_pytest.xml
PYTEST_RESULTS=$?
set -e

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports-${REAL_ARCH}.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
upload_artifact ${WORKSPACE_TMP}/test_reports-${REAL_ARCH}.tar.bz

TEST_RESULTS=$(($CTEST_RESULTS+$PYTEST_RESULTS))
exit ${TEST_RESULTS}
