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

# Set the codecov args used by all flags
CODECOV_ARGS="--root ${MRC_ROOT} --branch ${GITHUB_REF_NAME} --no-gcov-out --disable gcov"

# Add the PR if we are in a PR branch
if [[ "${GITHUB_REF_NAME}" =~ pull-request/[0-9]+ ]]; then
  CODECOV_ARGS="${CODECOV_ARGS} --pr ${GITHUB_REF_NAME##*/}"
fi

echo "CODECOV_ARGS: ${CODECOV_ARGS}"

rapids-logger "Running C++ Tests"
cd ${MRC_ROOT}/build

# Ensure we have a clean slate
find . -type f  \( -iname "*.gcov" -or -iname "*.gcda" \) -exec rm {} \;

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

rapids-logger "Compiling coverage for C++ tests"
cd ${MRC_ROOT}/build

# Run gcov manually. This is necessary since gcovr will only save the outputs to
# temp directories. Adding the source-prefix enables codecov to function
# correctly and enabling relative only ignores system and conda files.
find . -type f -name '*.gcda' -exec x86_64-conda_cos6-linux-gnu-gcov -pbc --source-prefix ${MRC_ROOT} --relative-only {} + 1> /dev/null

rapids-logger "Uploading codecov for C++ tests"

# Get the list of files that we are interested in (Keeps the upload small)
GCOV_FILES=$(find . -type f \( -iname "cpp#mrc#include#*.gcov" -or -iname "python#*.gcov" -or -iname "cpp#mrc#src#*.gcov" \))

# Upload the .gcov files directly to codecov. They do a good job at processing the partials
/opt/conda/envs/mrc/bin/codecov ${CODECOV_ARGS} -f ${GCOV_FILES} -F cpp

# Remove the gcov files and any gcda files to reset counters
find . -type f  \( -iname "*.gcov" -or -iname "*.gcda" \) -exec rm {} \;

rapids-logger "Running Python Tests"
cd ${MRC_ROOT}/build/python

set +e
pytest -v --junit-xml=${WORKSPACE_TMP}/report_pytest.xml
PYTEST_RESULTS=$?
set -e

rapids-logger "Compiling coverage for Python tests"
cd ${MRC_ROOT}/build

# Run gcov manually. This is necessary since gcovr will only save the outputs to
# temp directories. Adding the source-prefix enables codecov to function
# correctly and enabling relative only ignores system and conda files.
find . -type f -name '*.gcda' -exec x86_64-conda_cos6-linux-gnu-gcov -pbc --source-prefix ${MRC_ROOT} --relative-only {} + 1> /dev/null

rapids-logger "Uploading codecov for Python tests"

# Get the list of files that we are interested in (Keeps the upload small)
GCOV_FILES=$(find . -type f \( -iname "cpp#mrc#include#*.gcov" -or -iname "python#*.gcov" -or -iname "cpp#mrc#src#*.gcov" \))

# Upload the .gcov files directly to codecov. They do a good job at processing the partials
/opt/conda/envs/mrc/bin/codecov ${CODECOV_ARGS} -f ${GCOV_FILES} -F py

# Remove the gcov files and any gcda files to reset counters
find . -type f  \( -iname "*.gcov" -or -iname "*.gcda" \) -exec rm {} \;

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}/"
aws s3 cp ${WORKSPACE_TMP}/test_reports.tar.bz "${ARTIFACT_URL}/test_reports.tar.bz"

TEST_RESULTS=$(($CTEST_RESULTS+$PYTEST_RESULTS))
exit ${TEST_RESULTS}
