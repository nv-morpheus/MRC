<!-- omit in toc -->
# Morpheus Runtime Core (MRC)

The **Morpheus Runtime Core** (MRC) library is a **reactive, network-aware, flexible, and performance-oriented streaming data library** that standardizes building modular and reusable pipelines with both C++ and Python. Designed to efficiently handle non-deterministic volumes and highly variable data streams, MRC is the backbone for NVIDIA's [Morpheus](https://github.com/nv-morpheus/Morpheus) cyber security library. It's goal is to serve as a common high-performance streaming data layer in which all personas of developers, ranging from Data Scientists to DevOps and Performance Engineers, find value.

<!-- omit in toc -->
### Major features and differentiators
 - Built in C++ for performance, with Python bindings for ease of use and rapid prototyping, with options for maximizing performance
 - Distributed computation with message transfers over RDMA using UCX
 - Dynamic reconfiguration to scale up and out at runtime​; requires no changes to pipeline configuration
 - Unopinionated data model: messages of any type can be used in the pipeline
 - Built from the ground up with asynchronous computation for mitigation of I/O and GPU blocking
 - Automatically handles backpressure (when the sender is operating faster than the receiver can keep up) and reschedules computation as needed

<!-- omit in toc -->
### Anatomy of a MRC Pipeline

![MRC Pipeline](docs/imgs/mrc_pipeline.jpg)

<!-- omit in toc -->
## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Conda Installation](#conda-installation)
    - [Python Bindings](#python-bindings)
    - [C++ Bindings](#c-bindings)
    - [Full MRC Library](#full-mrc-library)
    - [Optional Settings](#optional-settings)
  - [Source Installation](#source-installation)
  - [Docker Installation](#docker-installation)
- [Quickstart Guide](#quickstart-guide)
- [Contributing](#contributing)
  - [Third-Party code](#third-party-code)


## Installation
MRC includes both Python and C++ bindings and supports installation via [conda](https://docs.conda.io/en/latest/), Docker, or from source.

### Prerequisites

- Volta architecture (Compute capability 7.0) or better
- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- [conda or miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- If using Docker:
  - [Docker](https://docs.docker.com/get-docker/)
  - [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Conda Installation
Installing via Conda is the easiest method for getting the MRC components and supports both the Python and C++ bindings. To install the MRC conda package and build the C++ and hybrid components, follow the steps below. Please note, Conda performance can be greatly increased via [Mamba](https://github.com/mamba-org/mamba). To install Mamba in an existing Conda environment, simply run:

```bash
conda install -c conda-forge mamba
```

If you choose to use Mamba rather than Conda, simply replace `conda` with `mamba` in the instructions below.

#### Python Bindings
```bash
# If needed, create a new conda environment
conda create --name mrc python=3.10

# Activate the newly created conda environment
conda activate mrc

# Install MRC Python bindings
conda install -c rapidsai -c nvidia -c conda-forge mrc
```

#### C++ Bindings
```bash
# If needed, create a new conda environment
conda create --name mrc python=3.10

# Activate the newly created conda environment
conda activate mrc

# Install MRC Python bindings
conda install -c rapidsai -c nvidia -c conda-forge libmrc
```

#### Full MRC Library
```bash
# If needed, create a new conda environment
conda create --name mrc python=3.10

# Activate the newly created conda environment
conda activate mrc

# Install MRC Python bindings
conda install -c rapidsai -c nvidia -c conda-forge mrc libmrc
```

#### Optional Settings
To avoid specifying the channels in the Conda commands above:
```bash
conda config --env --add channels conda-forge &&\
conda config --env --add channels nvidia &&\
conda config --env --add channels rapidsai
```

And to opt-in to nightly releases:
```bash
conda config --env --add channels nvidia/label/dev &&
conda config --env --add channels rapidsai-nightly
```

### Source Installation
Installing via source is for more advanced users and is necessary to try MRC features before an official release.

<!-- omit in toc -->
#### Clone MRC repository
```bash
export MRC_ROOT=$(pwd)/mrc
git clone --recurse-submodules git@github.com:nv-morpheus/mrc.git $MRC_ROOT
cd $MRC_ROOT
```
<!-- omit in toc -->
#### Create MRC Conda Environment
```bash
# note: `mamba` may be used in place of `conda` for better performance.
conda env create -n mrc-dev --file $MRC_ROOT/conda/environments/all_cuda-128_arch-$(arch).yaml
conda activate mrc-dev
```
<!-- omit in toc -->
#### Build MRC
```bash
mkdir $MRC_ROOT/build
cd $MRC_ROOT/build
cmake ..
make -j $(nproc)
```
<!-- omit in toc -->
#### Run MRC C++ Tests
```bash
export MRC_TEST_INTERNAL_DATA_PATH=$MRC_ROOT/cpp/mrc/src/tests
$MRC_ROOT/build/cpp/mrc/src/tests/test_mrc_private.x
$MRC_ROOT/build/cpp/mrc/tests/test_mrc.x
$MRC_ROOT/build/cpp/mrc/tests/logging/test_mrc_logging.x
```
<!-- omit in toc -->
#### Install MRC Python Bindings
```bash
pip install -e $MRC_ROOT/build/python
```
<!-- omit in toc -->
#### Run MRC Python Tests
```bash
pytest $MRC_ROOT/python
```

### Docker Installation
A Dockerfile is provided at `$MRC_ROOT` and can be built with
```bash
DOCKER_BUILDKIT=1 docker build -t mrc:latest .
```
To run the container
```bash
docker run --gpus all --cap-add=sys_nice -v $PWD:/work --rm -it mrc:latest /bin/bash
```

> **Note:**
> Users wishing to debug MRC in a Docker container should add the following to the `docker run` command:
> `--cap-add=SYS_PTRACE`

## Quickstart Guide

To quickly learn about both the C++ and Python MRC APIs, including following along with various complexity examples, we recommend following the MRC Quickstart Repository located [here](/docs/quickstart/README.md). This tutorial walks new users through topics like
- Creating a simple MRC pipeline
- Using a custom datatype between MRC nodes
- Using Reactive-style operators inside nodes for complex functionality
- Understand how threads and buffers can effect performance
- Mixing C++ and Python, for example - defining compute nodes in C++ and calling them from Python

## Contributing
To learn how to contribute to MRC, please read the [Contribution Guide](CONTRIBUTING.md).

MRC is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

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

### Third-Party code
Thirdparty code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a url to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the url to the original code.

Ex:
```
/*
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
