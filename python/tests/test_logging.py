# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from unittest import mock

from srf.core import log_handler as srf_log_handler
from srf.core import logging as srf_logging


@mock.patch('srf.core.logging.log')
def test_logging(mock_srf_log):
    assert not srf_logging.is_initialized()
    assert srf_logging.init_logging("log test", logging.ERROR)
    assert srf_logging.is_initialized()
    assert srf_logging.get_level() == logging.ERROR

    # Calling init_logging a second time is a noop
    assert not srf_logging.init_logging("log test")

    srf_logging.set_level(logging.INFO)
    assert srf_logging.get_level() == logging.INFO

    handler = srf_log_handler.SrfHandler()
    handler.setLevel(logging.WARNING)

    assert srf_logging.get_level() == logging.WARNING

    logger = logging.getLogger()
    logger.addHandler(handler)

    # logger has a lower level than our handler
    logger.setLevel(logging.INFO)

    logger.info('test info')

    mock_srf_log.assert_not_called()
    logger.error('test error')

    mock_srf_log.assert_called_once()
