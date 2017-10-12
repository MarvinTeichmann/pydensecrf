"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(0, "../../")

from pydensecrf import densecrf

import pytest


def test_exp_and_normalize():
    np_matrix = np.random.randn(3, 3).astype(np.float32)
    result = densecrf.py_expAndNormalize(np_matrix)
    return result


if __name__ == '__main__':
    logging.info("Hello World.")
