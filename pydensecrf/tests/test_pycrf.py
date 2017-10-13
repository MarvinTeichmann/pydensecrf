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


def _get_simple_unary():
    unary1 = np.zeros((10, 10), dtype=np.float32)
    unary1[:, [0, -1]] = unary1[[0, -1], :] = 1

    unary2 = np.zeros((10, 10), dtype=np.float32)
    unary2[4:7, 4:7] = 1

    unary = np.vstack([unary1.flat, unary2.flat])
    unary = (unary + 1) / (np.sum(unary, axis=0) + 2)

    return unary


def test_unary_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    pcrf = pycrf.DenseCRF(100, 2)

    unary = _get_simple_unary()
    dcrf.setUnaryEnergy(-np.log(unary))
    pcrf.set_unary_energy(-np.log(unary))

    cresult = dcrf.inference(5)
    pyresult = pcrf.inference(5)

    assert(np.all(np.array(cresult) == pyresult))


if __name__ == '__main__':
    logging.info("Hello World.")
