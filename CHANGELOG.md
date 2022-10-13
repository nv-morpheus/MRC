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
