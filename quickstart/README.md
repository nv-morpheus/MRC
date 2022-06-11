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

## Python Quickstart

To get started with the SRF Python Runtime, there are several examples located in the `quickstart/examples/python` directory. These examples are organized into separate folders each showing a different topic. Each example directory has a name with the format, `XX_${EXAMPLE_NAME}`, where `XX` represents the example number (in increasing difficulty) and `${EXAMPLE_NAME}` is the example name. Below is a list of the available examples and a brief description:

| #      | Name | Description |
| ----------- | ----------- | --- |
| 00 | SimplePipeline | A small, basic pipeline with only a single source and single sink |
| 01 | ThreeNodePipeline | |
| 02 | CustomTypes | |

### Setup

Install the SRF Python Libraries via Conda using the following:

```bash
conda install -c nvidia/label/dev srf
```

### Running the Examples

Each example directory contains a `README.md` file with information about the example and a `run.py` python file. To run any of the examples, simply launch the `run.py` file from python:

```bash
python quickstart/examples/python/XX_ExampleName/run.py
```


## C++ Quickstart

### Conda Build

- Clone the repository and navigate to the quickstart folder

```
git clone https://github.com/nv-morpheus/srf.git
cd srf/quickstart
```

- Install the Conda Environment

```
conda env create -n srf-quickstart -f environment_cpp.yml
conda activate srf-quickstart
```

- Build
  - creating a library (srf_ext_quickstart) with a single SRF Source, see `include` and `src` directories
  - compile an example program `quickstart.x` that uses our `IntSource` and provides a basic sink to a `srf::Pipeline`

```
mkdir build
cd build
cmake ..
make
```

Execute
```
./examples/cpp/00_Quickstart/quickstart.x
```

Output
```
srf pipeline starting...
srf pipeline complete: counter should be 3; counter=3
```
