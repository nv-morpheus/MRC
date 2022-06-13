# SRF Quickstart

There are two basic ways to get started with SRF:
- install from the conda repositories
- pull the NGC container (available next release)

#### Prerequisites

- Pascal architecture or better
- NVIDIA driver `450.80.02` or higher
- [conda or miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- if using docker:
  - [Docker](https://docs.docker.com/get-docker/)
  - [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

# Python Quickstart

To get started with the SRF Python Runtime, there are several examples located in the `quickstart/examples/python` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `XX_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing difficulty) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | SimplePipeline | A small, basic pipeline with only a single source and single sink |
| 01 | ThreeNodePipeline | |
| 02 | CustomTypes | |

## Setup

Install the SRF Python Libraries via Conda using the following:

```bash
conda install -c nvidia/label/dev srf
```

## Running the Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python quickstart/examples/python/XX_ExampleName/run.py
```


# C++ Quickstart

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | SimplePipeline | A small, basic pipeline with a source, a node and a sink |
| 01 | libsrf_quickstart | A library with predefined C++ SRF nodes |
| 02 | PipelineWithLibrary | Pipeline using both library defined and locally defined nodes |

## Conda Build

### Clone the repository and navigate to the quickstart folder

```
git clone https://github.com/nv-morpheus/srf.git
cd srf/docs/quickstart/cpp
```

### Install the Conda Environment

```
conda env create -n srf-quickstart -f environment_cpp.yml
conda activate srf-quickstart
```

### Build

```
mkdir build
cd build
cmake ..
make
```


