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
from pydensecrf.tests import utils

import pytest


def test_unary_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    pcrf = pycrf.DenseCRF(100, 2)

    unary = utils._get_simple_unary()
    dcrf.setUnaryEnergy(-np.log(unary))
    pcrf.set_unary_energy(-np.log(unary))

    cresult = dcrf.inference(5)
    pyresult = pcrf.inference(5)

    assert(np.all(np.array(cresult) == pyresult))


if __name__ == '__main__':
    logging.info("Hello World.")
