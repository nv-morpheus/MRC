# Contributing to SRF

Contributions to SRF fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/NVIDIA/SRF/issues/new)
    describing in detail the problem or new feature. The SRF team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/NVIDIA/SRF/issues/new). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by SRF's code of conduct.
More information can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to look for issues with the [good first issue](https://github.com/NVIDIA/SRF/issues) label.
2. Comment on the issue stating that you are going to work on it.
3. Code! Make sure to update unit tests! Ensure the [license headers are set properly](#Licensing).
4. When done, [create your pull request](https://github.com/NVIDIA/SRF/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a SRF developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

## Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can look at the prioritized issues for our next release in our [project boards](https://github.com/NVIDIA/SRF/projects).

> **Pro Tip:** Always look at the release board with the highest number for issues to work on. This is where SRF developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable contributing. Start with _Step 2_ above, commenting on the issue to let others know you are working on it. If you have any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

### Build in a Conda Environment

#### Clone SRF repository
```
export $SRF_HOME=$(pwd)/srf
git clone git@github.com:nv-morpheus/srf.git $SRF_HOME
cd $SRF_HOME
```

#### Create SRF Conda environment
```
mamba env create -n srf --file $SRF_HOME/ci/conda/environments/dev_env.yml
conda activate srf
```

#### Build SRF
```
mkdir $SRF_HOME/build
cd $SRF_HOME/build
cmake ..
make -j $(nproc)
```

#### Run SRF C++ Tests
```
$SRF_HOME/build/tests/test_srf.x
$SRF_HOME/build/tests/logging/test_srf_logging.x
```

### Install SRF Python
```
pip install -e $SRF_HOME/python
```

#### Run SRF Python Tests
```
pytest $SRF_HOME/python
```

## Licensing
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


---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md \
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
