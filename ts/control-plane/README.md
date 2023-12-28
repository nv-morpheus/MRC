## Build Instructions

Process is very manually currenly. Follow these steps:

1. Update the conda environment: `mamba env update -n ${CONDA_DEFAULT_ENV} -f ./ci/conda/environments/dev_env.yml`
2. Change directory to the node folder: `cd ts/control-plane/`
3. Install the node dependencies: `npm install`
4. Build the proto files: `npm run build:proto`
5. Build the server files: `npm run build:server`
6. Run the tests: `npm run test`


Note: When working with the C++ tests, it will try to start a local server. To connect to an already running control
plane (i.e. for debugging), use the following environment variable:

```
export MRC_ARCHITECT_URL="localhost:13337"
```

## Debugging Instructions

### Pre-requisites

The following steps need to be completed before debugging. They only need to be performed once. If there are changes to the proto files, `package.json` or `tsconfig.json`, these steps will need to be repeated.

1. Update the conda environment: `mamba env update -n ${CONDA_DEFAULT_ENV} -f ./ci/conda/environments/dev_env.yml`
2. Change directory to the node folder: `cd ts/control-plane/`
3. Install the node dependencies: `npm install`
4. Build the proto files: `npm run build:proto`

### Start Session

To spin up a debugging session, follow the steps below:

1. Change directory to the node folder: `cd ts/control-plane/`
2. In you C++ debugging session window, set the variable `export MRC_SKIP_LAUNCH_NODE=1` to prevent the C++ tests from starting a new server automatically.
   1. Often you may need to start a debugging session first for the terminal to appear in VS Code.
3. Start the server
   1. Run `npm run start:server:debug`
   2. This will start the server with redux devtools enabled. You can browse to `http://localhost:9000` to view the redux devtools.
4. Run C++ Tests
   1. Click into the testing tab in VS Code
   2. Click the debug icon next to any test
   3. This will start the tests and wait for a debugger to attach.
   4. You should see the Redux Devtools update once a connection is made.
