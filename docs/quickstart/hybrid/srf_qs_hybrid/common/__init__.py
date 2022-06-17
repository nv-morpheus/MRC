import logging
import os

import srf
from srf.core.log_handler import SrfHandler

# Intentionally import everything from the pybind11 module into this package
from .data import *  # NOQA
from .nodes import *  # NOQA


def setup_logger(example_file: str):

    # Setup logging
    srf.logging.init_logging(os.path.split(os.path.dirname(example_file))[1], logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(SrfHandler())
    logger.setLevel(logging.INFO)

    return logger
