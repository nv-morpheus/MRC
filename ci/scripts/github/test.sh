#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
fetch_s3 "${ARTIFACT_ENDPOINT}/dot_cache.tar.bz" "${WORKSPACE_TMP}/dot_cache.tar.bz"
fetch_s3 "${ARTIFACT_ENDPOINT}/build.tar.bz" "${WORKSPACE_TMP}/build.tar.bz"

tar xf "${WORKSPACE_TMP}/dot_cache.tar.bz"
tar xf "${WORKSPACE_TMP}/build.tar.bz"

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

rapids-logger "Installing SRF"
cmake -P ${SRF_ROOT}/build/cmake_install.cmake
pip install ${SRF_ROOT}/build/python

if [[ "${BUILD_CC}" == "gcc-coverage" ]]; then
  CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_BUILD_WITH_CODECOV}"
else
  CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
fi

cmake -B build -G Ninja ${CMAKE_FLAGS} .

rapids-logger "Running C++ Tests"
cd ${SRF_ROOT}/build
set +e
# Tests known to be failing
# Issues:
# * test_srf_private - https://github.com/nv-morpheus/SRF/issues/33
# * nvrpc - https://github.com/nv-morpheus/SRF/issues/34
ctest --output-on-failure \
      --exclude-regex "test_srf_private|nvrpc" \
      --output-junit ${REPORTS_DIR}/report_ctest.xml

CTEST_RESULTS=$?
set -e
cd ${SRF_ROOT}

rapids-logger "Running Python Tests"
cd ${SRF_ROOT}/build/python
set +e
pytest -v --junit-xml=${WORKSPACE_TMP}/report_pytest.xml
PYTEST_RESULTS=$?
set -e

if [[ "${BUILD_CC}" == "gcc-coverage" ]]; then
  rapids-logger "Generating codecov report"
  cd ${SRF_ROOT}
  cmake --build build --target gcovr-html-report gcovr-xml-report

  rapids-logger "Archiving codecov report"
  tar cfj ${WORKSPACE_TMP}/coverage_reports.tar.bz ${SRF_ROOT}/build/gcovr-html-report
  aws s3 cp ${WORKSPACE_TMP}/coverage_reports.tar.bz "${ARTIFACT_URL}/coverage_reports.tar.bz"

  gpuci_logger "Upload codecov report"
  codecov --root ${SRF_ROOT} -f ${SRF_ROOT}/build/gcovr-xml-report.xml
fi

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp ${WORKSPACE_TMP}/test_reports.tar.bz "${ARTIFACT_URL}/test_reports.tar.bz"

TEST_RESULTS=$(($CTEST_RESULTS+$PYTEST_RESULTS))
exit ${TEST_RESULTS}
