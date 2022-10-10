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

restore_conda_env

gpuci_logger "Fetching Build artifacts from ${DISPLAY_ARTIFACT_URL}/"
fetch_s3 "${ARTIFACT_ENDPOINT}/cpp_tests.tar.bz" "${WORKSPACE_TMP}/cpp_tests.tar.bz"
fetch_s3 "${ARTIFACT_ENDPOINT}/dsos.tar.bz" "${WORKSPACE_TMP}/dsos.tar.bz"
fetch_s3 "${ARTIFACT_ENDPOINT}/python_build.tar.bz" "${WORKSPACE_TMP}/python_build.tar.bz"

tar xf "${WORKSPACE_TMP}/cpp_tests.tar.bz"
tar xf "${WORKSPACE_TMP}/dsos.tar.bz"
tar xf "${WORKSPACE_TMP}/python_build.tar.bz"

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

# ctest requires cmake to be configured in order to locate tests

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
  cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} ${CMAKE_BUILD_WITH_CODECOV} .
else
  cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} .
fi

gpuci_logger "Running C++ Tests"
cd ${SRF_ROOT}/build
set +e
# Tests known to be failing
# Issues:
# * test_srf_benchmarking - https://github.com/nv-morpheus/SRF/issues/32
# * test_srf_private - https://github.com/nv-morpheus/SRF/issues/33
# * nvrpc - https://github.com/nv-morpheus/SRF/issues/34
ctest --output-on-failure \
      --exclude-regex "test_srf_private|nvrpc" \
      --output-junit ${REPORTS_DIR}/report_ctest.xml

CTEST_RESULTS=$?
set -e
cd ${SRF_ROOT}

gpuci_logger "Running Python Tests"
cd ${SRF_ROOT}/build/python
set +e
pytest -v --junit-xml=${WORKSPACE_TMP}/report_pytest.xml
PYTEST_RESULTS=$?
set -e

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
  gpuci_logger "Generating codecov report"
  cd ${SRF_ROOT}
  cmake --build build --target gcovr-html-report

  gpuci_logger "Archiving codecov report"
  tar cfj ${WORKSPACE_TMP}/coverage_reports.tar.bz ${SRF_ROOT}/build/gcovr-html-report
  aws s3 cp ${WORKSPACE_TMP}/coverage_reports.tar.bz "${ARTIFACT_URL}/coverage_reports.tar.bz"
fi

gpuci_logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp ${WORKSPACE_TMP}/test_reports.tar.bz "${ARTIFACT_URL}/test_reports.tar.bz"

TEST_RESULTS=$(($CTEST_RESULTS+$PYTEST_RESULTS))
exit ${TEST_RESULTS}
