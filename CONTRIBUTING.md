# Contributing to MRC

Contributions to MRC fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/NVIDIA/MRC/issues/new)
    describing in detail the problem or new feature. The MRC team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/NVIDIA/MRC/issues/new). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by MRC's code of conduct.
More information can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to look for issues with the [good first issue](https://github.com/NVIDIA/MRC/issues) label.
2. Comment on the issue stating that you are going to work on it.
3. Code! Make sure to update unit tests and confirm that test coverage has not decreased (see below)! Ensure the
[license headers are set properly](#Licensing).
4. When done, [create your pull request](https://github.com/NVIDIA/MRC/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, an MRC developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

## Unit testing and Code Coverage
Prior to submitting a pull request, you should ensure that all your contributed code is covered by unit tests, and that
unit test coverage percentages have not decreased (even better if they've increased). To test, from the MRC root
directory:

1. Generate a code coverage report and ensure your additions are covered.
   1. Take note of the CUDA Toolkit setup in the Build section below
   2. `./scripts/gen_coverage.sh`
   3. open `./build/gcovr-html-report/index.html`

## Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can look at the prioritized issues for our next release in our [project boards](https://github.com/NVIDIA/MRC/projects).

> **Pro Tip:** Always look at the release board with the highest number for issues to work on. This is where MRC developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable contributing. Start with _Step 2_ above, commenting on the issue to let others know you are working on it. If you have any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

### Build in a Conda Environment

#### CUDA Toolkit Setup

MRC uses the CUDA Toolkit. If you have multiple versions installed on your host, then some care needs to be taken in your environment.
For example, you may see the following error when activating the mrc conda environment:

`Version of installed CUDA didn't match package`

Some options:

- Set the variable `CUDA_HOME` to the desired CUDA install
  - This option is good for overriding the value set in `PATH` if you have multiple installs
  - This will also get rid of the warning messages when activating conda
  - Note: This must be set before calling `conda activate` and will only work for the lifetime of that shell session. For that reason, it's best to configure this in your `.bashrc` or similar configuration file.

- Set the CMake CUDA variable `-DCUDAToolkit_ROOT`
  - For example, you can set `-DCUDAToolkit_ROOT=/usr/local/cuda-11.5` to tell CMake to use your CUDA 11.5 install
  - This will work even if the `nvcc_linux-64` conda package is uninstalled

#### Clone MRC repository
```bash
export MRC_ROOT=$(pwd)/mrc
git clone --recurse-submodules git@github.com:nv-morpheus/mrc.git $MRC_ROOT
cd $MRC_ROOT
```

#### Create MRC Conda environment
```bash
# note: `mamba` may be used in place of `conda` for better performance.
conda env create -n mrc --file $MRC_ROOT/conda/environments/all_cuda-128_arch-x86_64.yaml
conda activate mrc
```
#### Build MRC
```bash
mkdir $MRC_ROOT/build
cd $MRC_ROOT/build
cmake ..
make -j $(nproc)
```

#### Run MRC C++ Tests
```bash
export MRC_TEST_INTERNAL_DATA_PATH=$MRC_ROOT/cpp/mrc/src/tests
$MRC_ROOT/build/cpp/mrcsrc/tests/test_mrc_private.x
$MRC_ROOT/build/cpp/mrctests/test_mrc.x
$MRC_ROOT/build/cpp/mrctests/logging/test_mrc_logging.x
```

### Install MRC Python
```bash
pip install -e $MRC_ROOT/build/python
```

#### Run MRC Python Tests
```bash
pytest $MRC_ROOT/python
```

### Building API Documentation
From the root of the MRC repo, configure CMake with `MRC_BUILD_DOCS=ON` then build the `mrc_docs` target. Once built the documentation will be located in the `build/docs/html` directory.
```bash
cmake -B build -DMRC_BUILD_DOCS=ON .
cmake --build build --target mrc_docs
```

## Licensing
MRC is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```c++
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
```c++
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


---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md \
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
