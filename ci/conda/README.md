# Conda Packaging

To build the Conda packages, it's recommended to run the provided scripts from a docker container. To build the container, `srf-conda-build`, run the following:

```bash
cd ${SRF_HOME}
docker buildx build --target development -t srf-conda-build .
```

This will create the image `srf-conda-build` that can be used to build SRF conda packages. When running this container, is recommended to set the environment variable `CONDA_PKGS_DIRS` to a path mounted on the host to speed up the build process. Without this variable set, the packages needed during the build will need to be re-downloaded each time the container is run.

## Building the Conda Package Locally

To build and save the SRF conda package, run the following:

```bash
docker run --rm -ti -v $PWD:/work \
   -e CONDA_PKGS_DIRS=/work/.cache/conda_pkgs \
   -e CONDA_ARGS="--output-folder=/work/.conda-bld" \
   srf-conda-build ./ci/conda/recipes/run_conda_build.sh
```

This will save the conda packages to `${SRF_HOME}/.conda-bld`. To install from this location, use the following:

```bash
conda install -c file://${SRF_HOME}/.conda-bld srf
```

## Uploading the Conda Package

To upload the conda package, run the following:

```bash
docker run --rm -ti -v $PWD:/work \
   -e CONDA_PKGS_DIRS=/work/.cache/conda_pkgs \
   -e CONDA_TOKEN=${CONDA_TOKEN:?"CONDA_TOKEN must be set to allow upload"} \
   srf-conda-build ./ci/conda/recipes/run_conda_build.sh upload
```

**Note:** This is only for internal SRF developers and will fail if you do not have the correct upload token.
