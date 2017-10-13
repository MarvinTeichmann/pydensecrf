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

from pydensecrf import densecrf
from pydensecrf import py_densecrf as pycrf

import pytest


def test_exp_and_normalize():
    np_matrix = np.random.randn(3, 3).astype(np.float32)
    result = densecrf.exp_and_normalize(np_matrix)
    return result


def test_eq_exp_and_normalize():
    np_matrix = np.random.randn(2, 4).astype(np.float32)
    cresult = densecrf.exp_and_normalize(np_matrix)
    pyresult = pycrf.exp_and_normalize(np_matrix)
    assert(np.all(cresult == pyresult))


if __name__ == '__main__':
    logging.info("Hello World.")
