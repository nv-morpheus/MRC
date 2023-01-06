# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from mrc.core import log_handler as mrc_log_handler
from mrc.core import logging as mrc_logging


@mock.patch('mrc.core.logging.log')
def test_logging(mock_mrc_log, is_debugger_attached: bool):

    # Because we want to initialize the logger for our own testing purposes, and because glog its not possible to
    # uninitialize the logger, we will just verify that its already initialized and test the other functions
    assert mrc_logging.is_initialized()
    assert mrc_logging.get_level() == logging.INFO if is_debugger_attached else logging.WARNING

    # Calling init_logging a second time is a noop
    assert not mrc_logging.init_logging("log test")

    mrc_logging.set_level(logging.ERROR)
    assert mrc_logging.get_level() == logging.ERROR

    handler = mrc_log_handler.MrcHandler()
    handler.setLevel(logging.WARNING)

    assert mrc_logging.get_level() == logging.WARNING

    logger = logging.getLogger()
    logger.addHandler(handler)

    # logger has a lower level than our handler
    logger.setLevel(logging.INFO)

    logger.info('test info')

    mock_mrc_log.assert_not_called()
    logger.error('test error')

    mock_mrc_log.assert_called_once()
