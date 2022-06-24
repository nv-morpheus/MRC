# SRF Quick Start Guide (QSG)

The SRF Quick Start Guide (QSG) provides examples on how to start using SRF via the Python bindings, C++ bindings or both.

## Prerequisites

- Pascal architecture (Compute capability 6.0) or better
- NVIDIA driver `450.80.02` or higher
- [conda or miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
  - **Note:** Conda performance can be improved by using [Mamba](https://github.com/mamba-org/mamba) and is recommended for the QSG. If you have `mamba` installed, simply replace `conda`  with `mamba` in the installation instructions.
- if using docker:
  - [Docker](https://docs.docker.com/get-docker/)
  - [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Getting Started

To get started with the QSG, it is necessary to get the required SRF components before running any of the examples. The QSG supports two methods for getting the SRF components: installing via Conda and building from source.

### Installing via Conda [preferred]
Installing via Conda is the easiest method for getting the SRF components and supports both the Python and C++ bindings. To install the SRF conda package and build the C++ and Hybrid components, follow the steps below:

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

### Building from Source [advanced]
Installing via the source is for more advanced users and is necessary to try SRF features before an official release. To install SRF from source, follow the [build instructions](../../CONTRIBUTING.md#setting-up-your-build-environment) in the [CONTRIBUTING](./../CONTRIBUTING.md) guide.

**Note:** When building with this method, the CMake configure option `-DSRF_BUILD_DOCS:BOOL=ON` will build and install all of the necessary packages to run every example in the QSG during the source file compilation. If this option is used, the "Setup" steps in the Python, C++ and Hybrid sections below can be skipped.

## Python Quickstart

For users interested in using SRF from Python, the QSG provides several examples in the `docs/quickstart/python/srf_qs_python` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [simple_pipeline](./python/srf_qs_python/ex00_simple_pipeline/README.md) | A small, basic pipeline with only a single source, node and sink |
| 01 | [custom_data](./python/srf_qs_python/ex01_custom_data/README.md) | Similar to simple_pipeline, but passes a custom data type between nodes |
| 02 | [reactive_operators](./python/srf_qs_python/ex02_reactive_operators/README.md) | Demonstrates how to use Reactive style operators inside of nodes for more complex functionality |
| 03 | [config_options](./python/srf_qs_python/ex03_config_options/README.md) | Illustrates how thread and buffer options can alter performance |

### Setup

Before starting with any of the examples, it's necessary to install the `srf_qs_python` package into your current conda environment.

Note: This section can be skipped if `-DSRF_BUILD_DOCS:BOOL=ON` was included in the "Getting Started" -> "Build from Source" section.

To install the python `srf_qs_python` package, run the following command:

```bash
# Change directory to the repo root
cd ${SRF_HOME}

# Pip install the package
pip install -e docs/quickstart/python
```

Once installed, the examples can be run from any directory.

### Running the Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python docs/quickstart/python/srf_qs_python/**ex##_ExampleName**/run.py
```

Some examples have configurable options to alter the behavior of the example. To see what options are available, pass `--help` to the example's `run.py` file. For example:

```bash
$ python docs/quickstart/python/srf_qs_python/ex03_config_options/run.py --help
usage: run.py [-h] [--count COUNT] [--channel_size CHANNEL_SIZE] [--threads THREADS]

ConfigOptions Example.

optional arguments:
  -h, --help            show this help message and exit
  --count COUNT         The number of items for the source to emit
  --channel_size CHANNEL_SIZE
                        The size of the inter-node buffers. Must be a power of 2
  --threads THREADS     The number of threads to use.
```


## C++ Quickstart

For users interested in using SRF with C++, the QSG provides several examples in the `docs/quickstart/cpp` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [simple_pipeline](./cpp/ex00_simple_pipeline/README.md) | A small, basic pipeline with only a single source, node and sink |
| 01 | [node_library](./cpp/ex01_node_library/README.md) | Illustrates hopw to create SRF components in a reusable library |
| 02 | [pipeline_with_library](./cpp/ex02_pipeline_with_library/README.md) | Demonstrates how to use the SRF components from Example #01 in a SRF Pipeline |

### Setup

Before starting with any of the examples, it's necessary to build the C++ examples.

Note: This section can be skipped if `-DSRF_BUILD_DOCS:BOOL=ON` was included in the "Getting Started" -> "Build from Source" section.

To build the C++ examples, run the following command:

```bash
# Change directory to the repo root
cd ${SRF_HOME}/docs/quickstart

# Compile the C++ examples
./compile.sh
```

This will output all built C++ examples in the `${SRF_HOME}/docs/quickstart/build` folder which will be referred to as `BUILD_DIR`.

### Running the Examples

Each example directory contains a `README.md` file with information about the example. To run any of the examples, follow the instructions in the `README.md` for launching the example.

## Hybrid Quickstart

For users interested in using SRF in a hybrid environment with both C++ and Python, the QSG provides several examples in the `docs/quickstart/hybrid/srf_qs_hybrid` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [wrap_data_objects](./hybrid/srf_qs_hybrid/ex00_wrap_data_objects/README.md) | How to run a pipeline in Python using C++ data objects |
| 01 | [wrap_nodes](./hybrid/srf_qs_hybrid/ex01_wrap_nodes/README.md) | How to run a pipeline in Python using sources, sinks, and nodes defined in C++ |
| 02 | [mixed_execution](./hybrid/srf_qs_hybrid/ex02_mixed_execution/README.md) | How to run a pipeline with some nodes in python and others in C++ |

### Setup

Before starting with any of the examples, it's necessary to install the `srf_qs_hybrid` package into your current conda environment.

Note: This section can be skipped if `-DSRF_BUILD_DOCS:BOOL=ON` was included in the "Getting Started" -> "Build from Source" section.

To install the python `srf_qs_hybrid` package, run the following command:

```bash
# Change directory to the repo root
cd ${SRF_HOME}/docs/quickstart

# Compile the C++ examples
./compile.sh
```

Once installed, the examples can be run from any directory.

### Running the Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python docs/quickstart/hybrid/srf_qs_python/<ex##_ExampleName>/run.py
```

Some examples have configurable options to alter the behavior of the example. To see what options are available, pass `--help` to the example's `run.py` file. For example:

```bash
$ python docs/quickstart/hybrid/srf_qs_hybrid/ex02_mixed_execution/run.py --help
usage: run.py [-h] [--python_source] [--python_node] [--python_sink]

mixed_execution Example.

optional arguments:
  -h, --help       show this help message and exit
  --python_source  Specifying this argument will run the pipeline with a python source
  --python_node    Specifying this argument will run the pipeline with a python node
  --python_sink    Specifying this argument will run the pipeline with a python sink
```
