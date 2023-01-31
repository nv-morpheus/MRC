<!-- omit in toc -->
# MRC Quick Start Guide

The MRC Quick Start Guide (QSG) provides Python, C++, and Hybrid (Python and C++) examples on how to get started building high-performance, streaming data pipelines with the MRC library. The goal of this tutorial is to provide a gentle on-ramp to building applications with MRC and is targeted towards a wide persona of developers - ranging from data scientists to performance engineers.

<!-- omit in toc -->
## Table of Contents
- [Environment Setup](#environment-setup)
  - [Installing Examples via Conda](#installing-examples-via-conda)
  - [Installing Examples via Source](#installing-examples-via-source)
- [Python MRC Examples](#python-mrc-examples)
- [C++ MRC Examples](#c-mrc-examples)
- [Hybrid (Python and C++) MRC Examples](#hybrid-python-and-c-mrc-examples)


## Environment Setup

To get started with the Quick Start Guide, it is necessary to install the required MRC components and dependencies before running any of the examples. Similar to the MRC library, this tutorial supports installation either via Conda (preferred) or by building from source.

### Installing Examples via Conda

If you've alredy followed the MRC library installation instructions in the [main project README](../../README.md#installation), you've most likely already created a Conda environment from which either `mrc` (Python) or `libmrc` (C++) was installed. In this case, you can install the QSG dependencies and build the examples with:

```bash
# Change directory to the quickstart root
cd ${MRC_HOME}/docs/quickstart/

# Update the existing MRC conda environment. Here, we assume the environment is named `mrc`, but you can change this to the environment name you used, if different
conda env update -n mrc -f environment_cpp.yml
conda activate mrc

# Build the QSG including the C++ examples
./compile.sh

# Install Python module (mrc_qs_python) to build the MRC QSG Python examples
pip install -e python
```

If you haven't installed MRC or would like to create an entirely new Conda environment for the Quick Start Guide, run:
```bash
# Change directory to the quickstart root
cd ${MRC_HOME}/docs/quickstart/

# Create a new MRC conda environment. Here, we assume the environment is named `mrc-quickstart`
conda env create -n mrc-quickstart -f environment_cpp.yml
conda activate mrc-quickstart

# Build the QSG including the C++ examples
./compile.sh

# Install Python module (mrc_qs_python) to build the MRC QSG Python examples
pip install -e python
```

**Note**:
If you encounter errors building the examples, this is mostly likely caused for two reasons:
1. You forgot to activate your Conda environment prior to running `./compile.sh`
2. There's a stale build lingering which needs to be cleared with `rm -rf build` before running `./compile.sh` again

### Installing Examples via Source

A comprehensive overview of building MRC from source is provided in the [MRC README](../../README.md#source-installation). To build the Quick Start Guide examples, simply run

```bash
cd ${MRC_HOME}/docs/quickstart

# Build the QSG including the C++ examples
./compile.sh

# Install Python module (mrc_qs_python) to build the MRC QSG Python examples
pip install -e python
```

## Python MRC Examples

For users interested in using MRC from Python, the QSG provides several examples in the `docs/quickstart/python/mrc_qs_python` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [simple_pipeline](./python/mrc_qs_python/ex00_simple_pipeline/README.md) | A small, basic pipeline with only a single source, node and sink |
| 01 | [custom_data](./python/mrc_qs_python/ex01_custom_data/README.md) | Similar to simple_pipeline, but passes a custom data type between nodes |
| 02 | [reactive_operators](./python/mrc_qs_python/ex02_reactive_operators/README.md) | Demonstrates how to use Reactive style operators inside of nodes for more complex functionality |
| 03 | [config_options](./python/mrc_qs_python/ex03_config_options/README.md) | Illustrates how thread and buffer options can alter performance |

Once installed, the examples can be run from any directory.

<!-- omit in toc -->
### Running the Python MRC Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` Python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python docs/quickstart/python/mrc_qs_python/**ex##_ExampleName**/run.py
```

Some examples have configurable options to alter the behavior of the example. To see what options are available, pass `--help` to the example's `run.py` file. For example:

```bash
$ python docs/quickstart/python/mrc_qs_python/ex03_config_options/run.py --help
usage: run.py [-h] [--count COUNT] [--channel_size CHANNEL_SIZE] [--threads THREADS]

ConfigOptions Example.

optional arguments:
  -h, --help            show this help message and exit
  --count COUNT         The number of items for the source to emit
  --channel_size CHANNEL_SIZE
                        The size of the inter-node buffers. Must be a power of 2
  --threads THREADS     The number of threads to use.
```

## C++ MRC Examples

For users interested in using MRC with C++, the QSG provides several examples in the `docs/quickstart/cpp` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [simple_pipeline](./cpp/ex00_simple_pipeline/README.md) | A small, basic pipeline with only a single source, node and sink |
| 01 | [node_library](./cpp/ex01_node_library/README.md) | Illustrates hopw to create MRC components in a reusable library |
| 02 | [pipeline_with_library](./cpp/ex02_pipeline_with_library/README.md) | Demonstrates how to use the MRC components from Example #01 in a MRC Pipeline |

<!-- omit in toc -->
### Running the C++ MRC Examples

Each example directory contains a `README.md` file with information about the example. To run any of the examples, follow the instructions in the `README.md` for launching the example.

## Hybrid (Python and C++) MRC Examples

For users interested in using MRC in a hybrid environment with both C++ and Python, the QSG provides several examples in the `docs/quickstart/hybrid/mrc_qs_hybrid` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `ex##_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing complexity) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | [wrap_data_objects](./hybrid/mrc_qs_hybrid/ex00_wrap_data_objects/README.md) | How to run a pipeline in Python using C++ data objects |
| 01 | [wrap_nodes](./hybrid/mrc_qs_hybrid/ex01_wrap_nodes/README.md) | How to run a pipeline in Python using sources, sinks, and nodes defined in C++ |
| 02 | [mixed_execution](./hybrid/mrc_qs_hybrid/ex02_mixed_execution/README.md) | How to run a pipeline with some nodes in python and others in C++ |

<!-- omit in toc -->
### Running the Hybrid MRC Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python docs/quickstart/hybrid/mrc_qs_python/<ex##_ExampleName>/run.py
```

Some examples have configurable options to alter the behavior of the example. To see what options are available, pass `--help` to the example's `run.py` file. For example:

```bash
$ python docs/quickstart/hybrid/mrc_qs_hybrid/ex02_mixed_execution/run.py --help
usage: run.py [-h] [--python_source] [--python_node] [--python_sink]

mixed_execution Example.

optional arguments:
  -h, --help       show this help message and exit
  --python_source  Specifying this argument will run the pipeline with a python source
  --python_node    Specifying this argument will run the pipeline with a python node
  --python_sink    Specifying this argument will run the pipeline with a python sink
```
