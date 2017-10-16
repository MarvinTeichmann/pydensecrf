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

import pydensecrf.pairwise as pair

from pydensecrf import densecrf

import pydensecrf.utils as utils
from pydensecrf.tests import utils as test_utils

from pydensecrf.py_densecrf import exp_and_normalize


def test_pairwise():
    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()
    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)
    pairwise = pair.PairwisePotentials(feats, compat=3)
    out = pairwise.apply(unary)

    return out


def compute_inference_with_pair(plist, lg_unary, num_iter):
    prediction = exp_and_normalize(-lg_unary)
    for i in range(num_iter):
        tmp1 = -lg_unary
        for potential in plist:
            tmp2 = potential.apply(prediction)
            tmp1 = tmp1 - tmp2
        prediction = exp_and_normalize(tmp1)
    return prediction


def test_pairwise_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    dcrf.setUnaryEnergy(-np.log(unary))
    lg_unary = -np.log(unary)

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    dcrf.addPairwiseEnergy(feats, compat=3)
    pairwise = pair.PairwisePotentials(feats, compat=3)

    dres = np.argmax(dcrf.inference(10), axis=0).reshape(10, 10)
    out = compute_inference_with_pair([pairwise], lg_unary, 10)
    pres = np.argmax(out, axis=0).reshape(10, 10)

    assert(np.all(dres == pres))


def compute_inference_with_dkernel(klist, clist, lg_unary, num_iter):
    prediction = exp_and_normalize(-lg_unary)
    for i in range(num_iter):
        tmp1 = -lg_unary
        for kernel, comp in zip(klist, clist):
            tmp2 = kernel.apply(prediction)
            tmp2 = comp.apply(tmp2)
            tmp1 = tmp1 - tmp2
        prediction = exp_and_normalize(tmp1)
    return prediction


def test_dkernel_inference():

    dcrf = densecrf.DenseCRF(100, 2)
    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    dcrf.setUnaryEnergy(-np.log(unary))
    lg_unary = -np.log(unary)

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    dcrf.addPairwiseEnergy(feats, compat=3)
    klist = [pair.DenseKernel(feats)]
    clist = [pair.PottsComp(3)]

    dres = np.argmax(dcrf.inference(10), axis=0).reshape(10, 10)
    out = compute_inference_with_dkernel(klist, clist, lg_unary, 10)
    pres = np.argmax(out, axis=0).reshape(10, 10)

    assert(np.all(dres == pres))
