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

REPORTS_DIR="${WORKSPACE_TMP}/reports"

BENCHMARKS=($(find ${SRF_ROOT}/build/benchmarks -name "*.x"))

rapids-logger "Running Benchmarks..."
BENCH_RESULTS=0
for benchmark in "${BENCHMARKS[@]}"; do
       bench_name=$(basename ${benchmark})
       rapids-logger "Running ${bench_name}"
       set +e

       taskset -c 0 ${benchmark} --benchmark_out_format=json --benchmark_out="${REPORTS_DIR}/${bench_name}.json"
       BENCH_RESULT=$?
       rapids-logger "${bench_name} completed with an exit status of ${BENCH_RESULT}"
       BENCH_RESULTS=$(($BENCH_RESULTS+$BENCH_RESULT))

       set -e
done

# We want the archive step to run, even if we failed, save our exit status, which the next step can use
echo ${BENCH_RESULTS} > ${WORKSPACE_TMP}/exit_status

rapids-logger "Running Benchmarks... Done"
