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


if __name__ == '__main__':
    logging.info("Hello World.")


from utils import unary_from_labels, create_pairwise_bilateral
from utils import create_pairwise_gaussian


class DenseCRF():
    """This is a reimplementation of DenseCRF (almost) entirely in python.
    """
    def __init__(self, npixels, nclasses):
        super(DenseCRF, self).__init__()
        self.npixels = npixels
        self.nclasses = nclasses

    def set_unary_energy(self, unary):
        self.unary = unary
        return

    def add_pairwise_energy(self, feats, compat=3,
                            kernel=None, normalization=None):
        pass

    def inference(self, num_iter=5):
        pass

    def start_inference():
        pass

    def step_inference():
        pass
