<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# MRC 24.06.00 (03 Jul 2024)

## 🚀 New Features

- Add JSONValues container for holding Python values as JSON objects if possible, and as pybind11::object otherwise ([#455](https://github.com/nv-morpheus/MRC/pull/455)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🛠️ Improvements

- resolve rapids-dependency-file-generator warning ([#482](https://github.com/nv-morpheus/MRC/pull/482)) [@jameslamb](https://github.com/jameslamb)
- Downgrade doxygen to match Morpheus ([#469](https://github.com/nv-morpheus/MRC/pull/469)) [@cwharris](https://github.com/cwharris)
- Consolidate redundant split_string_to_array, split_string_on &amp; split_path methods ([#465](https://github.com/nv-morpheus/MRC/pull/465)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add pybind11 type caster for JSONValues ([#458](https://github.com/nv-morpheus/MRC/pull/458)) [@dagardner-nv](https://github.com/dagardner-nv)

# MRC 24.03.01 (16 Apr 2024)

## 🐛 Bug Fixes

- Add auto register helpers to AsyncSink and AsyncSource ([#473](https://github.com/nv-morpheus/MRC/pull/473)) [@dagardner-nv](https://github.com/dagardner-nv)

# MRC 24.03.00 (7 Apr 2024)

## 🚨 Breaking Changes

- Update cast_from_pyobject to throw on unsupported types rather than returning null ([#451](https://github.com/nv-morpheus/MRC/pull/451)) [@dagardner-nv](https://github.com/dagardner-nv)
- RAPIDS 24.02 Upgrade ([#433](https://github.com/nv-morpheus/MRC/pull/433)) [@cwharris](https://github.com/cwharris)

## 🐛 Bug Fixes

- Update CR year ([#460](https://github.com/nv-morpheus/MRC/pull/460)) [@dagardner-nv](https://github.com/dagardner-nv)
- Removing the INFO log when creating an AsyncioRunnable ([#456](https://github.com/nv-morpheus/MRC/pull/456)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update cast_from_pyobject to throw on unsupported types rather than returning null ([#451](https://github.com/nv-morpheus/MRC/pull/451)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt updated builds of CI runners ([#442](https://github.com/nv-morpheus/MRC/pull/442)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update Conda channels to prioritize `conda-forge` over `nvidia` ([#436](https://github.com/nv-morpheus/MRC/pull/436)) [@cwharris](https://github.com/cwharris)
- Remove redundant copy of libmrc_pymrc.so ([#429](https://github.com/nv-morpheus/MRC/pull/429)) [@dagardner-nv](https://github.com/dagardner-nv)
- Unifying cmake exports name across all Morpheus repos ([#427](https://github.com/nv-morpheus/MRC/pull/427)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Updating the workspace settings to remove deprecated python options ([#425](https://github.com/nv-morpheus/MRC/pull/425)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Use `dependencies.yaml` to generate environment files ([#416](https://github.com/nv-morpheus/MRC/pull/416)) [@cwharris](https://github.com/cwharris)

## 📖 Documentation

- Update minimum requirements ([#467](https://github.com/nv-morpheus/MRC/pull/467)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Add maximum simultaneous tasks support to `TaskContainer` ([#464](https://github.com/nv-morpheus/MRC/pull/464)) [@cwharris](https://github.com/cwharris)
- Add `TestScheduler` to support testing time-based coroutines without waiting for timeouts ([#453](https://github.com/nv-morpheus/MRC/pull/453)) [@cwharris](https://github.com/cwharris)
- Adding RoundRobinRouter node type for distributing values to downstream nodes ([#449](https://github.com/nv-morpheus/MRC/pull/449)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add IoScheduler to enable epoll-based Task scheduling ([#448](https://github.com/nv-morpheus/MRC/pull/448)) [@cwharris](https://github.com/cwharris)
- Update ops-bot.yaml ([#446](https://github.com/nv-morpheus/MRC/pull/446)) [@AyodeAwe](https://github.com/AyodeAwe)
- RAPIDS 24.02 Upgrade ([#433](https://github.com/nv-morpheus/MRC/pull/433)) [@cwharris](https://github.com/cwharris)

## 🛠️ Improvements

- Update MRC to use CCCL instead of libcudacxx ([#444](https://github.com/nv-morpheus/MRC/pull/444)) [@cwharris](https://github.com/cwharris)
- Optionally skip the CI pipeline if the PR contains the skip-ci label ([#426](https://github.com/nv-morpheus/MRC/pull/426)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add flake8, yapf, and isort pre-commit hooks. ([#420](https://github.com/nv-morpheus/MRC/pull/420)) [@cwharris](https://github.com/cwharris)

# MRC 23.11.00 (30 Nov 2023)

## 🐛 Bug Fixes

- Use a traditional semaphore in AsyncioRunnable ([#412](https://github.com/nv-morpheus/MRC/pull/412)) [@cwharris](https://github.com/cwharris)
- Fix libhwloc &amp; stubgen versions to match dev yaml ([#405](https://github.com/nv-morpheus/MRC/pull/405)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update boost versions to match version used in dev env ([#404](https://github.com/nv-morpheus/MRC/pull/404)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix EdgeHolder from incorrectly reporting an active connection ([#402](https://github.com/nv-morpheus/MRC/pull/402)) [@dagardner-nv](https://github.com/dagardner-nv)
- Safe handling of control plane promises &amp; fix CI ([#391](https://github.com/nv-morpheus/MRC/pull/391)) [@dagardner-nv](https://github.com/dagardner-nv)
- Revert boost upgrade, and update clang to v16 ([#382](https://github.com/nv-morpheus/MRC/pull/382)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fixing an issue with `update-versions.sh` which always blocked CI ([#377](https://github.com/nv-morpheus/MRC/pull/377)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add test for  gc being invoked in a thread finalizer ([#365](https://github.com/nv-morpheus/MRC/pull/365)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt patched pybind11 ([#364](https://github.com/nv-morpheus/MRC/pull/364)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Add missing flags to docker command to mount the working dir and set -cap-add=sys_nice ([#383](https://github.com/nv-morpheus/MRC/pull/383)) [@dagardner-nv](https://github.com/dagardner-nv)
- Make Quick Start Guide not use `make_node_full` ([#376](https://github.com/nv-morpheus/MRC/pull/376)) [@cwharris](https://github.com/cwharris)

## 🚀 New Features

- Add AsyncioRunnable ([#411](https://github.com/nv-morpheus/MRC/pull/411)) [@cwharris](https://github.com/cwharris)
- Adding more coroutine components to support async generators and task containers ([#408](https://github.com/nv-morpheus/MRC/pull/408)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update ObservableProxy::pipe to support any number of operators ([#387](https://github.com/nv-morpheus/MRC/pull/387)) [@cwharris](https://github.com/cwharris)
- Updates for MRC/Morpheus to build in the same RAPIDS devcontainer environment ([#375](https://github.com/nv-morpheus/MRC/pull/375)) [@cwharris](https://github.com/cwharris)

## 🛠️ Improvements

- Move Pycoro from Morpheus to MRC ([#409](https://github.com/nv-morpheus/MRC/pull/409)) [@cwharris](https://github.com/cwharris)
- update rapidsai/ci to rapidsai/ci-conda ([#396](https://github.com/nv-morpheus/MRC/pull/396)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add local CI scripts &amp; rebase docker image ([#394](https://github.com/nv-morpheus/MRC/pull/394)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use `copy-pr-bot` ([#369](https://github.com/nv-morpheus/MRC/pull/369)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update Versions for v23.11.00 ([#357](https://github.com/nv-morpheus/MRC/pull/357)) [@mdemoret-nv](https://github.com/mdemoret-nv)

# MRC 23.07.00 (19 Jul 2023)

## 🚨 Breaking Changes

- Remove `mrc::internals` namespace and cleanup class names ([#328](https://github.com/nv-morpheus/MRC/pull/328)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Switch to using `cuda-toolkit` over `cudatoolkit` ([#320](https://github.com/nv-morpheus/MRC/pull/320)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update to Python 3.10 ([#317](https://github.com/nv-morpheus/MRC/pull/317)) [@cwharris](https://github.com/cwharris)

## 🐛 Bug Fixes

- Fixing actions running on non-PR branches ([#354](https://github.com/nv-morpheus/MRC/pull/354)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix the conda build with RMM 23.02 ([#348](https://github.com/nv-morpheus/MRC/pull/348)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Workaround GCC 11.3 compiler bug ([#339](https://github.com/nv-morpheus/MRC/pull/339)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- use correct ostream.h header location ([#333](https://github.com/nv-morpheus/MRC/pull/333)) [@cwharris](https://github.com/cwharris)
- Bug fix -- in some situations it was possible for persistent modules to overwrite each other ([#331](https://github.com/nv-morpheus/MRC/pull/331)) [@drobison00](https://github.com/drobison00)
- Release an RxNodeComponent edge on error ([#327](https://github.com/nv-morpheus/MRC/pull/327)) [@dagardner-nv](https://github.com/dagardner-nv)
- RxNodeComponent should set exceptions on the context ([#326](https://github.com/nv-morpheus/MRC/pull/326)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update python debug detection for new version of `debugpy` ([#322](https://github.com/nv-morpheus/MRC/pull/322)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix nullptr dereference in NvmlState ([#319](https://github.com/nv-morpheus/MRC/pull/319)) [@cwharris](https://github.com/cwharris)
- Dynamically loading `libnvidia-ml.so.1` instead of directly linking ([#313](https://github.com/nv-morpheus/MRC/pull/313)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- fix libabseil package name typo ([#310](https://github.com/nv-morpheus/MRC/pull/310)) [@cwharris](https://github.com/cwharris)

## 📖 Documentation

- Fix a few minor type-o&#39;s in comments ([#332](https://github.com/nv-morpheus/MRC/pull/332)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix license headers to only use a single /* comment to exclude it from doxygen ([#307](https://github.com/nv-morpheus/MRC/pull/307)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Use libgrpc=1.54 ([#353](https://github.com/nv-morpheus/MRC/pull/353)) [@cwharris](https://github.com/cwharris)
- Adding option to configure running the conda-build CI step with labels ([#349](https://github.com/nv-morpheus/MRC/pull/349)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Upgrade RMM version to 23.06 ([#346](https://github.com/nv-morpheus/MRC/pull/346)) [@cwharris](https://github.com/cwharris)
- Create label-external-issues.yml ([#323](https://github.com/nv-morpheus/MRC/pull/323)) [@jarmak-nv](https://github.com/jarmak-nv)
- Support RMM 22.12 with Python 3.8 ([#318](https://github.com/nv-morpheus/MRC/pull/318)) [@cwharris](https://github.com/cwharris)
- Update to Python 3.10 ([#317](https://github.com/nv-morpheus/MRC/pull/317)) [@cwharris](https://github.com/cwharris)
- Adding an `update-version.sh` script and CI check to keep versions up to date ([#314](https://github.com/nv-morpheus/MRC/pull/314)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update to RMM 23.02 ([#309](https://github.com/nv-morpheus/MRC/pull/309)) [@cwharris](https://github.com/cwharris)
- Devcontainer Updates ([#297](https://github.com/nv-morpheus/MRC/pull/297)) [@cwharris](https://github.com/cwharris)
- add git-lfs and gh config dir ([#273](https://github.com/nv-morpheus/MRC/pull/273)) [@cwharris](https://github.com/cwharris)

## 🛠️ Improvements

- New CI images with rapids 23.06 ([#351](https://github.com/nv-morpheus/MRC/pull/351)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove patch from pybind11 ([#335](https://github.com/nv-morpheus/MRC/pull/335)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove `boost::filesystem` ([#334](https://github.com/nv-morpheus/MRC/pull/334)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Remove `mrc::internals` namespace and cleanup class names ([#328](https://github.com/nv-morpheus/MRC/pull/328)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Switch to using `cuda-toolkit` over `cudatoolkit` ([#320](https://github.com/nv-morpheus/MRC/pull/320)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- GH Infra Updates: PR Template, Discussions, Add To Project Action ([#316](https://github.com/nv-morpheus/MRC/pull/316)) [@jarmak-nv](https://github.com/jarmak-nv)
- Use ARC V2 self-hosted runners for GPU jobs ([#315](https://github.com/nv-morpheus/MRC/pull/315)) [@jjacobelli](https://github.com/jjacobelli)
- Use newly built CI images with CUDA 11.8 ([#311](https://github.com/nv-morpheus/MRC/pull/311)) [@dagardner-nv](https://github.com/dagardner-nv)
- bump version to 23.07 ([#306](https://github.com/nv-morpheus/MRC/pull/306)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use ARC V2 self-hosted runners for CPU jobs ([#302](https://github.com/nv-morpheus/MRC/pull/302)) [@jjacobelli](https://github.com/jjacobelli)

# MRC 23.03.00 (29 Mar 2023)

## 🚨 Breaking Changes

- Cleanup top-level forward.hpp and types.hpp ([#292](https://github.com/nv-morpheus/MRC/pull/292)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🐛 Bug Fixes

- Cleanup top-level forward.hpp and types.hpp ([#292](https://github.com/nv-morpheus/MRC/pull/292)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🛠️ Improvements

- Updating to driver 525 ([#299](https://github.com/nv-morpheus/MRC/pull/299)) [@jjacobelli](https://github.com/jjacobelli)
- Improvements to the python module generation CMake code ([#298](https://github.com/nv-morpheus/MRC/pull/298)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update workflow `permissions` block ([#296](https://github.com/nv-morpheus/MRC/pull/296)) [@ajschmidt8](https://github.com/ajschmidt8)
- Set AWS credentials lifetime to 12h ([#295](https://github.com/nv-morpheus/MRC/pull/295)) [@jjacobelli](https://github.com/jjacobelli)
- Use AWS OIDC to get AWS creds ([#294](https://github.com/nv-morpheus/MRC/pull/294)) [@jjacobelli](https://github.com/jjacobelli)
- Pointer cast macro ([#293](https://github.com/nv-morpheus/MRC/pull/293)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update `sccache` bucket ([#289](https://github.com/nv-morpheus/MRC/pull/289)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update CMake to only add fcoroutines flag if clang version is less than 16 + bump to latest utils ([#288](https://github.com/nv-morpheus/MRC/pull/288)) [@drobison00](https://github.com/drobison00)
- Mirror module / buffer + python bindings. ([#286](https://github.com/nv-morpheus/MRC/pull/286)) [@drobison00](https://github.com/drobison00)
- Updating to use driver 520 ([#282](https://github.com/nv-morpheus/MRC/pull/282)) [@mdemoret-nv](https://github.com/mdemoret-nv)

# MRC 23.01.00 (30 Jan 2023)

## 🚨 Breaking Changes

- Non-Linear pipelines and Non-Runnable Nodes ([#261](https://github.com/nv-morpheus/MRC/pull/261)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update Versions for 23.01 ([#242](https://github.com/nv-morpheus/MRC/pull/242)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Upgrade to C++20 ([#230](https://github.com/nv-morpheus/MRC/pull/230)) [@ryanolson](https://github.com/ryanolson)
- MRC Rename ([#221](https://github.com/nv-morpheus/MRC/pull/221)) [@ryanolson](https://github.com/ryanolson)
- Distributed Runtime ([#218](https://github.com/nv-morpheus/MRC/pull/218)) [@ryanolson](https://github.com/ryanolson)

## 🐛 Bug Fixes

- Add handling for empty downstream sources ([#279](https://github.com/nv-morpheus/MRC/pull/279)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Removing TL Expected ([#278](https://github.com/nv-morpheus/MRC/pull/278)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix function wrappers with `functools.partial` ([#277](https://github.com/nv-morpheus/MRC/pull/277)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Reverting relative paths for git submodules ([#274](https://github.com/nv-morpheus/MRC/pull/274)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- rename morpheus_utils_configure_boost_* to use new utilities naming ([#272](https://github.com/nv-morpheus/MRC/pull/272)) [@cwharris](https://github.com/cwharris)
- Bump utilities version ([#269](https://github.com/nv-morpheus/MRC/pull/269)) [@drobison00](https://github.com/drobison00)
- Missing &lt;unordered_map&gt; header; caught by gcc12 ([#256](https://github.com/nv-morpheus/MRC/pull/256)) [@ryanolson](https://github.com/ryanolson)
- Fix potential race condition ([#248](https://github.com/nv-morpheus/MRC/pull/248)) [@ryanolson](https://github.com/ryanolson)
- codecov updates ([#233](https://github.com/nv-morpheus/MRC/pull/233)) [@dagardner-nv](https://github.com/dagardner-nv)
- Ensure interpreter is properly initialized, minor test fixes ([#220](https://github.com/nv-morpheus/MRC/pull/220)) [@drobison00](https://github.com/drobison00)

## 📖 Documentation

- update clone instructions to include --recurse-submodules ([#271](https://github.com/nv-morpheus/MRC/pull/271)) [@cwharris](https://github.com/cwharris)

## 🚀 New Features

- Non-Linear pipelines and Non-Runnable Nodes ([#261](https://github.com/nv-morpheus/MRC/pull/261)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- C++20 Coroutines ([#244](https://github.com/nv-morpheus/MRC/pull/244)) [@ryanolson](https://github.com/ryanolson)
- Development container image ([#236](https://github.com/nv-morpheus/MRC/pull/236)) [@ryanolson](https://github.com/ryanolson)
- Workspace Update to allow indentation in CMake argument blocks ([#234](https://github.com/nv-morpheus/MRC/pull/234)) [@ryanolson](https://github.com/ryanolson)
- Upgrade to C++20 ([#230](https://github.com/nv-morpheus/MRC/pull/230)) [@ryanolson](https://github.com/ryanolson)
- MRC Rename ([#221](https://github.com/nv-morpheus/MRC/pull/221)) [@ryanolson](https://github.com/ryanolson)
- Distributed Runtime ([#218](https://github.com/nv-morpheus/MRC/pull/218)) [@ryanolson](https://github.com/ryanolson)

## 🛠️ Improvements

- Move Expected/Error from internal -&gt; core ([#263](https://github.com/nv-morpheus/MRC/pull/263)) [@ryanolson](https://github.com/ryanolson)
- Core Concepts ([#262](https://github.com/nv-morpheus/MRC/pull/262)) [@ryanolson](https://github.com/ryanolson)
- Improved Task Lifecycle ([#259](https://github.com/nv-morpheus/MRC/pull/259)) [@ryanolson](https://github.com/ryanolson)
- Update Task destructor logic to conform to the standard ([#258](https://github.com/nv-morpheus/MRC/pull/258)) [@ryanolson](https://github.com/ryanolson)
- Initial Fiber and Coroutine Benchmarks ([#255](https://github.com/nv-morpheus/MRC/pull/255)) [@ryanolson](https://github.com/ryanolson)
- 23.01 Clang Format Update ([#254](https://github.com/nv-morpheus/MRC/pull/254)) [@ryanolson](https://github.com/ryanolson)
- Updating `CODEOWNERS` for new repo organization ([#250](https://github.com/nv-morpheus/MRC/pull/250)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update IWYU to suggest forward declarations ([#249](https://github.com/nv-morpheus/MRC/pull/249)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Remove draconian regex filter that eliminates entire tests from being run ([#246](https://github.com/nv-morpheus/MRC/pull/246)) [@ryanolson](https://github.com/ryanolson)
- lazy instantiation of the nvml singleton ([#243](https://github.com/nv-morpheus/MRC/pull/243)) [@ryanolson](https://github.com/ryanolson)
- Update Versions for 23.01 ([#242](https://github.com/nv-morpheus/MRC/pull/242)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- MRC changes related to utility consolidation ([#241](https://github.com/nv-morpheus/MRC/pull/241)) [@drobison00](https://github.com/drobison00)
- C++20 version of upcoming C++23 std::expected ([#239](https://github.com/nv-morpheus/MRC/pull/239)) [@ryanolson](https://github.com/ryanolson)
- C++ Reorganization ([#237](https://github.com/nv-morpheus/MRC/pull/237)) [@ryanolson](https://github.com/ryanolson)
- Update codeowners to MRC ([#235](https://github.com/nv-morpheus/MRC/pull/235)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Adds missing functionality to allow dynamic modules to forward their configuration to nested children. ([#224](https://github.com/nv-morpheus/MRC/pull/224)) [@drobison00](https://github.com/drobison00)
- Run CI on main branch ([#217](https://github.com/nv-morpheus/MRC/pull/217)) [@dagardner-nv](https://github.com/dagardner-nv)

# SRF 22.11.00 (18 Nov 2022)

## 🚨 Breaking Changes

- update abseil, grpc, and ucx versions for cuml compatibility ([#177](https://github.com/nv-morpheus/MRC/pull/177)) [@cwharris](https://github.com/cwharris)

## 🐛 Bug Fixes

- Fix throwing errors from `Executor.join_async()` ([#208](https://github.com/nv-morpheus/MRC/pull/208)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix help string for SRF_BUILD_DOCS ([#202](https://github.com/nv-morpheus/MRC/pull/202)) [@dagardner-nv](https://github.com/dagardner-nv)
- change pull_request to pull_request_target ([#201](https://github.com/nv-morpheus/MRC/pull/201)) [@jarmak-nv](https://github.com/jarmak-nv)
- Registered memory should be released before the UCX Context is destroyed ([#193](https://github.com/nv-morpheus/MRC/pull/193)) [@ryanolson](https://github.com/ryanolson)
- Fix tests so that the proper upstream build is used for the coverage test ([#192](https://github.com/nv-morpheus/MRC/pull/192)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updating SRF versions from 22.09 to 22.11 ([#191](https://github.com/nv-morpheus/MRC/pull/191)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixes &quot;Add new issue/PR to project&quot; action ([#189](https://github.com/nv-morpheus/MRC/pull/189)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fetch history and tags for package step ([#188](https://github.com/nv-morpheus/MRC/pull/188)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix CI deps ([#187](https://github.com/nv-morpheus/MRC/pull/187)) [@dagardner-nv](https://github.com/dagardner-nv)
- Emit the value before incrementing the iterator fixes ([#180](https://github.com/nv-morpheus/MRC/pull/180)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix returning of thread_binding attr ([#179](https://github.com/nv-morpheus/MRC/pull/179)) [@dagardner-nv](https://github.com/dagardner-nv)
- update abseil, grpc, and ucx versions for cuml compatibility ([#177](https://github.com/nv-morpheus/MRC/pull/177)) [@cwharris](https://github.com/cwharris)

## 📖 Documentation

- Add documentation on how to build the doxygen docs ([#183](https://github.com/nv-morpheus/MRC/pull/183)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Replacing SRF markdown templates with yml forms ([#200](https://github.com/nv-morpheus/MRC/pull/200)) [@jarmak-nv](https://github.com/jarmak-nv)

## 🛠️ Improvements

- Improve NVML + MIG Behavior ([#206](https://github.com/nv-morpheus/MRC/pull/206)) [@ryanolson](https://github.com/ryanolson)
- Add dockerfile for CI runners ([#199](https://github.com/nv-morpheus/MRC/pull/199)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add codecov upload ([#197](https://github.com/nv-morpheus/MRC/pull/197)) [@dagardner-nv](https://github.com/dagardner-nv)
- SRF Modules and Module Registry Implementation ([#196](https://github.com/nv-morpheus/MRC/pull/196)) [@drobison00](https://github.com/drobison00)
- Allow building build without GPU and without a driver ([#195](https://github.com/nv-morpheus/MRC/pull/195)) [@dagardner-nv](https://github.com/dagardner-nv)
- Switch to github actions ([#182](https://github.com/nv-morpheus/MRC/pull/182)) [@dagardner-nv](https://github.com/dagardner-nv)

# SRF 22.09.00 (30 Sep 2022)

## 📖 Documentation

- CONTRIBUTING updates for CUDA ([#159](https://github.com/nv-morpheus/MRC/pull/159)) [@pdmack](https://github.com/pdmack)

## 🛠️ Improvements

- Bump Versions 22.09 ([#174](https://github.com/nv-morpheus/MRC/pull/174)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add missing checks for YAPF_RETVAL &amp; PRAGMA_CHECK_RETVAL in CI ([#173](https://github.com/nv-morpheus/MRC/pull/173)) [@dagardner-nv](https://github.com/dagardner-nv)

# SRF 22.08.00 (7 Sep 2022)

## 🐛 Bug Fixes

- Update PortBuilder to Work with Types That Do Not Have a Default Constructor ([#165](https://github.com/nv-morpheus/MRC/pull/165)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix contributing guide build ([#139](https://github.com/nv-morpheus/MRC/pull/139)) [@cwharris](https://github.com/cwharris)
- fix faulty assumption about remote key sizes ([#137](https://github.com/nv-morpheus/MRC/pull/137)) [@ryanolson](https://github.com/ryanolson)
- Resolves issue-32, re-add stats watchers to Rx and Python nodes ([#130](https://github.com/nv-morpheus/MRC/pull/130)) [@drobison00](https://github.com/drobison00)
- Fix SRF Conda Upload ([#70](https://github.com/nv-morpheus/MRC/pull/70)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 📖 Documentation

- Adjust contrib instructions for pip install location ([#141](https://github.com/nv-morpheus/MRC/pull/141)) [@pdmack](https://github.com/pdmack)
- Update CONTRIBUTING.md ([#133](https://github.com/nv-morpheus/MRC/pull/133)) [@pdmack](https://github.com/pdmack)
- Typo fix in README.md ([#108](https://github.com/nv-morpheus/MRC/pull/108)) [@yuvaldeg](https://github.com/yuvaldeg)
- Refresh and Simplification of QSG README ([#100](https://github.com/nv-morpheus/MRC/pull/100)) [@awthomp](https://github.com/awthomp)

## 🚀 New Features

- Internal Runtime Query + CPP Checks ([#113](https://github.com/nv-morpheus/MRC/pull/113)) [@ryanolson](https://github.com/ryanolson)
- Data Plane - Initial P2P and RDMA Get ([#112](https://github.com/nv-morpheus/MRC/pull/112)) [@ryanolson](https://github.com/ryanolson)
- Network Options ([#111](https://github.com/nv-morpheus/MRC/pull/111)) [@ryanolson](https://github.com/ryanolson)
- Transient Pool ([#110](https://github.com/nv-morpheus/MRC/pull/110)) [@ryanolson](https://github.com/ryanolson)

## 🛠️ Improvements

- Bump versions 22.08 ([#166](https://github.com/nv-morpheus/MRC/pull/166)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Action to Add Issues/PRs to Project ([#155](https://github.com/nv-morpheus/MRC/pull/155)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add ability to specify port data type for known c++ types from Python ([#153](https://github.com/nv-morpheus/MRC/pull/153)) [@drobison00](https://github.com/drobison00)
- Fix CPP checks for CI ([#147](https://github.com/nv-morpheus/MRC/pull/147)) [@dagardner-nv](https://github.com/dagardner-nv)
- Code coverage integration in SRF ([#105](https://github.com/nv-morpheus/MRC/pull/105)) [@drobison00](https://github.com/drobison00)
- Add codable interface for python objects, (Ingress|Egress)Ports python bindings, and other elements required for multi-segment. ([#18](https://github.com/nv-morpheus/MRC/pull/18)) [@drobison00](https://github.com/drobison00)

# SRF 22.06.01 (4 Jul 2022)

## 🐛 Bug Fixes

- Fix `flatten()` Operator With Infinite Sources ([#117](https://github.com/nv-morpheus/MRC/pull/117)) [@mdemoret-nv](https://github.com/mdemoret-nv)

# SRF 22.06.00 (28 Jun 2022)

## 🐛 Bug Fixes

- Moving the python_module_tools before the SRF import ([#87](https://github.com/nv-morpheus/MRC/pull/87)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix for pipelines beginning before start is called ([#83](https://github.com/nv-morpheus/MRC/pull/83)) [@ryanolson](https://github.com/ryanolson)
- host_partition_id incorrectly set with multiple gpus present ([#79](https://github.com/nv-morpheus/MRC/pull/79)) [@ryanolson](https://github.com/ryanolson)
- Changing RAPIDS_VERSION to SRF_RAPIDS_VERSION ([#73](https://github.com/nv-morpheus/MRC/pull/73)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Apply PR #70 to Correct Branch ([#71](https://github.com/nv-morpheus/MRC/pull/71)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add pybind11-stubgen to the conda build ([#60](https://github.com/nv-morpheus/MRC/pull/60)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Improved Resource ownership ([#39](https://github.com/nv-morpheus/MRC/pull/39)) [@ryanolson](https://github.com/ryanolson)
- Improve error handling on pipeline::Controller updates ([#29](https://github.com/nv-morpheus/MRC/pull/29)) [@ryanolson](https://github.com/ryanolson)
- adding PortGraph; validate Pipeline definitions ([#28](https://github.com/nv-morpheus/MRC/pull/28)) [@ryanolson](https://github.com/ryanolson)

## 📖 Documentation

- SRF README Update - In Preparation of Public Release ([#93](https://github.com/nv-morpheus/MRC/pull/93)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Initial CONTRIBUTING.md ([#45](https://github.com/nv-morpheus/MRC/pull/45)) [@cwharris](https://github.com/cwharris)

## 🚀 New Features

- Improvements Needed for Quickstart ([#52](https://github.com/nv-morpheus/MRC/pull/52)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- General Cleanup ([#47](https://github.com/nv-morpheus/MRC/pull/47)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🛠️ Improvements

- adding pragma once to logging.h ([#41](https://github.com/nv-morpheus/MRC/pull/41)) [@ryanolson](https://github.com/ryanolson)
- pimpl IBuilder ([#40](https://github.com/nv-morpheus/MRC/pull/40)) [@ryanolson](https://github.com/ryanolson)
- Add two missing headers that caused clang compile errors. ([#31](https://github.com/nv-morpheus/MRC/pull/31)) [@drobison00](https://github.com/drobison00)
- Enable CI for SRF ([#24](https://github.com/nv-morpheus/MRC/pull/24)) [@dagardner-nv](https://github.com/dagardner-nv)
- Quickstart ([#20](https://github.com/nv-morpheus/MRC/pull/20)) [@ryanolson](https://github.com/ryanolson)
- adding node::Queue; refactoring SinkChannel for code reuse ([#1](https://github.com/nv-morpheus/MRC/pull/1)) [@ryanolson](https://github.com/ryanolson)
