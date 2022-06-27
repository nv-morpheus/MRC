# Streaming Reactive Framework (SRF)

The **Streaming Reactive Framework** (SRF) library (proununced "surf") is a **reactive, network-aware, flexible, and performance-oriented streaming data framework** that standardizes building modular and reusable pipelines with both C++ and Python​. The goal of SRF is to serve as a common streaming data layer in which all personas of developers - ranging from Data Scientists to DevOps and Performance Engineers can find value.

### Major features and differentiators
 - Built in C++ for performance, with Python bindings for ease of use and rapid prototyping, with options for maximizing performance
 - Distributed computation with message transfers over RMDA using UCX
 - Dynamic reconfiguration to scale up and out at runtime​; requires no changes to pipeline configuration
 - Unopinionated data model: messages of any type can be used in the pipeline
 - Built from the ground up with asynchronous computation for mitigation of I/O and GPU blocking
 - Automatically handles backpressure (when the sender is operating faster than the receiver can keep up) and reschedules computation as needed

### Anatomy of a SRF Pipeline

![SRF Pipeline](docs/imgs/srf_pipeline.png)


## Table of Contents
- [Streaming Reactive Framework (SRF)](#streaming-reactive-framework-srf)
    - [Major features and differentiators](#major-features-and-differentiators)
    - [Anatomy of a SRF Pipeline](#anatomy-of-a-srf-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Python and C++ Bindings with Conda](#python-and-c-bindings-with-conda)
  - [Quickstart](#quickstart)
  - [Contributing](#contributing)
    - [Thirdparty code](#thirdparty-code)


## Installation
SRF includes both Python and C++ bindings and supports installation via [conda](https://docs.conda.io/en/latest/), Docker, or source.]

### Prerequisites

- Pascal architecture (Compute capability 6.0) or better
- NVIDIA driver `450.80.02` or higher
- [conda or miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
  - **Note:** Conda performance can be improved by using [Mamba](https://github.com/mamba-org/mamba) and is recommended for the QSG. If you have `mamba` installed, simply replace `conda`  with `mamba` in the installation instructions.
- if using docker:
  - [Docker](https://docs.docker.com/get-docker/)
  - [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Python and C++ Bindings with Conda
Installing via Conda is the easiest method for getting the SRF components and supports both the Python and C++ bindings. To install the SRF conda package and build the C++ and hybrid components, follow the steps below:

```bash
# Change directory to the quickstart root
cd ${SRF_HOME}/docs/quickstart/

# If needed, create a new conda environment
conda env create -n srf-quickstart -f environment_cpp.yml
conda activate srf-quickstart

# Or if reusing a conda environment, ensure all dependencies are installed
conda env update -n srf-quickstart -f environment_cpp.yml
conda activate srf-quickstart

# Compile the QSG. This will build the C++ and Hybrid components in ./build. And install the Python and Hybrid libraries
./compile.sh
```

## Quickstart

To get started with SRF, see the SRF Quickstart Repository located [here](/docs/quickstart/README.md).

## Contributing
SRF is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```
/*
 * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 ```

### Thirdparty code
Thirdparty code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a url to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the url to the original code.

Ex:
```
/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Original Source: https://github.com/org/other_project
//
// Original License:
// ...
```
