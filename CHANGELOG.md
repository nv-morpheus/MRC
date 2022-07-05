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
