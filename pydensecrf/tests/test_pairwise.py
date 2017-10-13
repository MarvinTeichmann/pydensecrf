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

import pydensecrf.utils as utils
from pydensecrf.tests import utils as test_utils


def test_pairwise():
    img = test_utils._get_simple_img()
    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)
    pairwise = pair.PairwisePotentials(feats, compat=3)

    return pairwise
