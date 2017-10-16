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
import pydensecrf.utils as utils
from pydensecrf.tests import utils as test_utils

import pytest


def test_unary_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    pcrf = pycrf.DenseCRF(100, 2)

    unary = test_utils._get_simple_unary()
    dcrf.setUnaryEnergy(-np.log(unary))
    pcrf.set_unary_energy(-np.log(unary))

    cresult = dcrf.inference(5)
    pyresult = pcrf.inference(5)

    assert(np.all(np.array(cresult) == pyresult))


def test_complete_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    pcrf = pycrf.DenseCRF(100, 2)

    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    dcrf.setUnaryEnergy(-np.log(unary))
    pcrf.set_unary_energy(-np.log(unary))

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    dcrf.addPairwiseEnergy(feats, compat=3)

    pcrf.add_pairwise_energy(feats, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res1 = np.argmax(dcrf.inference(10), axis=0).reshape(10, 10)
    res2 = np.argmax(pcrf.inference(10), axis=0).reshape(10, 10)

    assert(np.all(res1 == res2))


def test_complete_inference2():

    dcrf = densecrf.DenseCRF(100, 2)
    pcrf = pycrf.DenseCRF(100, 2)

    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    dcrf.setUnaryEnergy(-np.log(unary))
    pcrf.set_unary_energy(-np.log(unary))

    feats = utils.create_pairwise_gaussian(sdims=(1.5, 1.5),
                                           shape=img.shape[:2])

    dcrf.addPairwiseEnergy(feats, compat=3)
    pcrf.add_pairwise_energy(feats, compat=3)

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    dcrf.addPairwiseEnergy(feats, compat=3)
    pcrf.add_pairwise_energy(feats, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res1 = np.argmax(dcrf.inference(10), axis=0).reshape(10, 10)
    res2 = np.argmax(pcrf.inference(10), axis=0).reshape(10, 10)

    if False:
        # Save the result for visual inspection

        import scipy as scp
        import scipy.misc

        scp.misc.imsave("test.png", res1)

    assert(np.all(res1 == res2))


if __name__ == '__main__':
    logging.info("Hello World.")
