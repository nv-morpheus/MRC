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

conda activate mrc

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

rapids-logger "Installing MRC"
cmake -P ${MRC_ROOT}/build/cmake_install.cmake
pip install ${MRC_ROOT}/build/python

rapids-logger "Running C++ Tests"
cd ${MRC_ROOT}/build
set +e
# Tests known to be failing
# Issues:
# * test_mrc_private - https://github.com/nv-morpheus/MRC/issues/33
# * nvrpc - https://github.com/nv-morpheus/MRC/issues/34
ctest --output-on-failure \
      --exclude-regex "test_mrc_private|nvrpc" \
      --output-junit ${REPORTS_DIR}/report_ctest.xml

CTEST_RESULTS=$?
set -e

CODECOV_ARGS="--root ${MRC_ROOT} --branch ${GITHUB_REF_NAME} --pr ${GITHUB_REF_NAME##*/} --no-gcov-out --disable gcov"
echo "CODECOV_ARGS: ${CODECOV_ARGS}"

cd ${MRC_ROOT}

rapids-logger "Compiling coverage for C++ tests"

# Run gcovr and delete the stats
gcovr -j ${PARALLEL_LEVEL} --gcov-executable x86_64-conda-linux-gnu-gcov --xml build/gcovr-xml-report-cpp.xml --xml-pretty -r ${MRC_ROOT} --object-directory "$PWD/build" \
  --exclude-unreachable-branches --exclude-throw-branches \
  -f '^include/.*' -f '^python/.*' -f '^src/.*' \
  -e '^python/srf/_pysrf/tests/.*' -e '^python/srf/tests/.*' -e '^src/tests/.*' \
  -d -s -k

rapids-logger "Uploading codecov for C++ tests"
cd ${MRC_ROOT}/build

# Get the list of files that we are interested in (Keeps the upload small)
GCOV_FILES=$(find . -type f \( -iname "^#include#*.gcov" -or -iname "^#python#*.gcov" -or -iname "^#src#*.gcov" \))
/opt/conda/envs/mrc/bin/codecov ${CODECOV_ARGS} -f ${GCOV_FILES} -F cpp
rm *.gcov

rapids-logger "Running Python Tests"
cd ${MRC_ROOT}/build/python
set +e
pytest -v --junit-xml=${WORKSPACE_TMP}/report_pytest.xml
PYTEST_RESULTS=$?
set -e

cd ${MRC_ROOT}

rapids-logger "Compiling coverage for Python tests"

# Need to rerun gcovr for the python code now
gcovr -j ${PARALLEL_LEVEL} --gcov-executable x86_64-conda-linux-gnu-gcov --xml build/gcovr-xml-report-py.xml --xml-pretty -r ${MRC_ROOT} --object-directory "$PWD/build" \
  --exclude-unreachable-branches --exclude-throw-branches \
  -f '^include/.*' -f '^python/.*' -f '^src/.*' \
  -e '^python/srf/_pysrf/tests/.*' -e '^python/srf/tests/.*' -e '^src/tests/.*' \
  -d -s -k

rapids-logger "Uploading codecov for Python tests"
cd ${MRC_ROOT}/build

# Get the list of files that we are interested in (Keeps the upload small)
GCOV_FILES=$(find . -type f \( -iname "^#include#*.gcov" -or -iname "^#python#*.gcov" -or -iname "^#src#*.gcov" \))

ls *.gcov
echo "GCOV_FILES: ${GCOV_FILES}"

/opt/conda/envs/mrc/bin/codecov ${CODECOV_ARGS} -f ${GCOV_FILES} -F py
# rm *.gcov
ls

rapids-logger "Archiving codecov report"
tar cfj ${WORKSPACE_TMP}/coverage_reports.tar.bz ${MRC_ROOT}/build/gcovr-xml-report-*.xml
aws s3 cp ${WORKSPACE_TMP}/coverage_reports.tar.bz "${ARTIFACT_URL}/coverage_reports.tar.bz"


rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp ${WORKSPACE_TMP}/test_reports.tar.bz "${ARTIFACT_URL}/test_reports.tar.bz"

TEST_RESULTS=$(($CTEST_RESULTS+$PYTEST_RESULTS))
exit ${TEST_RESULTS}
