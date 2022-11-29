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

import mrc.core.logging as mrc_logging


class MrcHandler(logging.Handler):
    """
    Forwards logs to MRC's logger
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            mrc_logging.log(msg, record.levelno, record.filename, record.lineno)
        except RecursionError:  # See issue 36272 https://bugs.python.org/issue36272
            raise
        except Exception:
            self.handleError(record)

    def setLevel(self, level):
        """
        Set the logging level of this handler and the mrc
        """
        # This could be a string so let the parent figure out how to validate it
        super().setLevel(level)
        mrc_logging.set_level(self.level)
