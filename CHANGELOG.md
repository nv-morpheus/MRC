# SRF 22.11.00 (18 Nov 2022)

## 🚨 Breaking Changes

- update abseil, grpc, and ucx versions for cuml compatibility ([#177](https://github.com/nv-morpheus/SRF/pull/177)) [@cwharris](https://github.com/cwharris)

## 🐛 Bug Fixes

- Fix throwing errors from `Executor.join_async()` ([#208](https://github.com/nv-morpheus/SRF/pull/208)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix help string for SRF_BUILD_DOCS ([#202](https://github.com/nv-morpheus/SRF/pull/202)) [@dagardner-nv](https://github.com/dagardner-nv)
- change pull_request to pull_request_target ([#201](https://github.com/nv-morpheus/SRF/pull/201)) [@jarmak-nv](https://github.com/jarmak-nv)
- Registered memory should be released before the UCX Context is destroyed ([#193](https://github.com/nv-morpheus/SRF/pull/193)) [@ryanolson](https://github.com/ryanolson)
- Fix tests so that the proper upstream build is used for the coverage test ([#192](https://github.com/nv-morpheus/SRF/pull/192)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updating SRF versions from 22.09 to 22.11 ([#191](https://github.com/nv-morpheus/SRF/pull/191)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixes &quot;Add new issue/PR to project&quot; action ([#189](https://github.com/nv-morpheus/SRF/pull/189)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fetch history and tags for package step ([#188](https://github.com/nv-morpheus/SRF/pull/188)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix CI deps ([#187](https://github.com/nv-morpheus/SRF/pull/187)) [@dagardner-nv](https://github.com/dagardner-nv)
- Emit the value before incrementing the iterator fixes ([#180](https://github.com/nv-morpheus/SRF/pull/180)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix returning of thread_binding attr ([#179](https://github.com/nv-morpheus/SRF/pull/179)) [@dagardner-nv](https://github.com/dagardner-nv)
- update abseil, grpc, and ucx versions for cuml compatibility ([#177](https://github.com/nv-morpheus/SRF/pull/177)) [@cwharris](https://github.com/cwharris)

## 📖 Documentation

- Add documentation on how to build the doxygen docs ([#183](https://github.com/nv-morpheus/SRF/pull/183)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- : Replacing SRF markdown templates with yml forms ([#200](https://github.com/nv-morpheus/SRF/pull/200)) [@jarmak-nv](https://github.com/jarmak-nv)

## 🛠️ Improvements

- Improve NVML + MIG Behavior ([#206](https://github.com/nv-morpheus/SRF/pull/206)) [@ryanolson](https://github.com/ryanolson)
- Add dockerfile for CI runners ([#199](https://github.com/nv-morpheus/SRF/pull/199)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add codecov upload ([#197](https://github.com/nv-morpheus/SRF/pull/197)) [@dagardner-nv](https://github.com/dagardner-nv)
- SRF Modules and Module Registry Implementation ([#196](https://github.com/nv-morpheus/SRF/pull/196)) [@drobison00](https://github.com/drobison00)
- Allow building build without GPU and without a driver ([#195](https://github.com/nv-morpheus/SRF/pull/195)) [@dagardner-nv](https://github.com/dagardner-nv)
- Switch to github actions ([#182](https://github.com/nv-morpheus/SRF/pull/182)) [@dagardner-nv](https://github.com/dagardner-nv)

# SRF 22.09.00 (30 Sep 2022)

## 📖 Documentation

- CONTRIBUTING updates for CUDA ([#159](https://github.com/nv-morpheus/SRF/pull/159)) [@pdmack](https://github.com/pdmack)

## 🛠️ Improvements

- Bump Versions 22.09 ([#174](https://github.com/nv-morpheus/SRF/pull/174)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add missing checks for YAPF_RETVAL &amp; PRAGMA_CHECK_RETVAL in CI ([#173](https://github.com/nv-morpheus/SRF/pull/173)) [@dagardner-nv](https://github.com/dagardner-nv)

# SRF 22.08.00 (7 Sep 2022)

## 🐛 Bug Fixes

- Update PortBuilder to Work with Types That Do Not Have a Default Constructor ([#165](https://github.com/nv-morpheus/SRF/pull/165)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix contributing guide build ([#139](https://github.com/nv-morpheus/SRF/pull/139)) [@cwharris](https://github.com/cwharris)
- fix faulty assumption about remote key sizes ([#137](https://github.com/nv-morpheus/SRF/pull/137)) [@ryanolson](https://github.com/ryanolson)
- Resolves issue-32, re-add stats watchers to Rx and Python nodes ([#130](https://github.com/nv-morpheus/SRF/pull/130)) [@drobison00](https://github.com/drobison00)
- Fix SRF Conda Upload ([#70](https://github.com/nv-morpheus/SRF/pull/70)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 📖 Documentation

- Adjust contrib instructions for pip install location ([#141](https://github.com/nv-morpheus/SRF/pull/141)) [@pdmack](https://github.com/pdmack)
- Update CONTRIBUTING.md ([#133](https://github.com/nv-morpheus/SRF/pull/133)) [@pdmack](https://github.com/pdmack)
- Typo fix in README.md ([#108](https://github.com/nv-morpheus/SRF/pull/108)) [@yuvaldeg](https://github.com/yuvaldeg)
- Refresh and Simplification of QSG README ([#100](https://github.com/nv-morpheus/SRF/pull/100)) [@awthomp](https://github.com/awthomp)

## 🚀 New Features

- Internal Runtime Query + CPP Checks ([#113](https://github.com/nv-morpheus/SRF/pull/113)) [@ryanolson](https://github.com/ryanolson)
- Data Plane - Initial P2P and RDMA Get ([#112](https://github.com/nv-morpheus/SRF/pull/112)) [@ryanolson](https://github.com/ryanolson)
- Network Options ([#111](https://github.com/nv-morpheus/SRF/pull/111)) [@ryanolson](https://github.com/ryanolson)
- Transient Pool ([#110](https://github.com/nv-morpheus/SRF/pull/110)) [@ryanolson](https://github.com/ryanolson)

## 🛠️ Improvements

- Bump versions 22.08 ([#166](https://github.com/nv-morpheus/SRF/pull/166)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Action to Add Issues/PRs to Project ([#155](https://github.com/nv-morpheus/SRF/pull/155)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add ability to specify port data type for known c++ types from Python ([#153](https://github.com/nv-morpheus/SRF/pull/153)) [@drobison00](https://github.com/drobison00)
- Fix CPP checks for CI ([#147](https://github.com/nv-morpheus/SRF/pull/147)) [@dagardner-nv](https://github.com/dagardner-nv)
- Code coverage integration in SRF ([#105](https://github.com/nv-morpheus/SRF/pull/105)) [@drobison00](https://github.com/drobison00)
- Add codable interface for python objects, (Ingress|Egress)Ports python bindings, and other elements required for multi-segment. ([#18](https://github.com/nv-morpheus/SRF/pull/18)) [@drobison00](https://github.com/drobison00)

# SRF 22.06.01 (4 Jul 2022)

## 🐛 Bug Fixes

- Fix `flatten()` Operator With Infinite Sources ([#117](https://github.com/nv-morpheus/SRF/pull/117)) [@mdemoret-nv](https://github.com/mdemoret-nv)

# SRF 22.06.00 (28 Jun 2022)

## 🐛 Bug Fixes

- Moving the python_module_tools before the SRF import ([#87](https://github.com/nv-morpheus/SRF/pull/87)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix for pipelines beginning before start is called ([#83](https://github.com/nv-morpheus/SRF/pull/83)) [@ryanolson](https://github.com/ryanolson)
- host_partition_id incorrectly set with multiple gpus present ([#79](https://github.com/nv-morpheus/SRF/pull/79)) [@ryanolson](https://github.com/ryanolson)
- Changing RAPIDS_VERSION to SRF_RAPIDS_VERSION ([#73](https://github.com/nv-morpheus/SRF/pull/73)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Apply PR #70 to Correct Branch ([#71](https://github.com/nv-morpheus/SRF/pull/71)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add pybind11-stubgen to the conda build ([#60](https://github.com/nv-morpheus/SRF/pull/60)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Improved Resource ownership ([#39](https://github.com/nv-morpheus/SRF/pull/39)) [@ryanolson](https://github.com/ryanolson)
- Improve error handling on pipeline::Controller updates ([#29](https://github.com/nv-morpheus/SRF/pull/29)) [@ryanolson](https://github.com/ryanolson)
- adding PortGraph; validate Pipeline definitions ([#28](https://github.com/nv-morpheus/SRF/pull/28)) [@ryanolson](https://github.com/ryanolson)

## 📖 Documentation

- SRF README Update - In Preparation of Public Release ([#93](https://github.com/nv-morpheus/SRF/pull/93)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Initial CONTRIBUTING.md ([#45](https://github.com/nv-morpheus/SRF/pull/45)) [@cwharris](https://github.com/cwharris)

## 🚀 New Features

- Improvements Needed for Quickstart ([#52](https://github.com/nv-morpheus/SRF/pull/52)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- General Cleanup ([#47](https://github.com/nv-morpheus/SRF/pull/47)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🛠️ Improvements

- adding pragma once to logging.h ([#41](https://github.com/nv-morpheus/SRF/pull/41)) [@ryanolson](https://github.com/ryanolson)
- pimpl IBuilder ([#40](https://github.com/nv-morpheus/SRF/pull/40)) [@ryanolson](https://github.com/ryanolson)
- Add two missing headers that caused clang compile errors. ([#31](https://github.com/nv-morpheus/SRF/pull/31)) [@drobison00](https://github.com/drobison00)
- Enable CI for SRF ([#24](https://github.com/nv-morpheus/SRF/pull/24)) [@dagardner-nv](https://github.com/dagardner-nv)
- Quickstart ([#20](https://github.com/nv-morpheus/SRF/pull/20)) [@ryanolson](https://github.com/ryanolson)
- adding node::Queue; refactoring SinkChannel for code reuse ([#1](https://github.com/nv-morpheus/SRF/pull/1)) [@ryanolson](https://github.com/ryanolson)
