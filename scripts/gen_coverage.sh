#!/usr/bin/env bash
set -x
set -e

cmake -DCMAKE_BUILD_TYPE=Debug -GNinja -DCMAKE_BUILD_TYPE=Debug -DSRF_WITH_CODECOV=ON -DSRF_BUILD_PYTHON=ON -DSRF_BUILD_TESTS=ON -B ./build
cmake --build ./build
pip install -e ./build/python
cd ./build && ctest && pytest ./build/python/tests
cmake --build ./build --target gcovr-html-report