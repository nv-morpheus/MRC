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

```
Coming soon...
````

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
conda activate  srf-quickstart
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
