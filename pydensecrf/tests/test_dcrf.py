import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from pydensecrf.tests import utils as test_utils

import pytest


def test_call_dcrf2d():

    d = dcrf.DenseCRF2D(10, 10, 2)

    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    # d.setUnaryEnergy(PyConstUnary(-np.log(Up)))

    d.addPairwiseBilateral(sxy=2, srgb=2, rgbim=img, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res = np.argmax(d.inference(10), axis=0).reshape(10, 10)

    np.all(res == img[:, :, 0] / 255)


def test_call_dcrf():

    d = dcrf.DenseCRF(100, 2)

    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    # d.setUnaryEnergy(PyConstUnary(-np.log(Up)))

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res = np.argmax(d.inference(10), axis=0).reshape(10, 10)

    np.all(res == img[:, :, 0] / 255)


def test_call_dcrf_eq_dcrf2d():

    d = dcrf.DenseCRF(100, 2)
    d2 = dcrf.DenseCRF2D(10, 10, 2)

    unary = test_utils._get_simple_unary()
    img = test_utils._get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    d2.setUnaryEnergy(-np.log(unary))

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=3)

    d2.addPairwiseBilateral(sxy=2, srgb=2, rgbim=img, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res1 = np.argmax(d.inference(10), axis=0).reshape(10, 10)
    res2 = np.argmax(d2.inference(10), axis=0).reshape(10, 10)

    assert(np.all(res1 == res2))


def test_compact_wrong():

    # Tests whether expection is indeed raised
    ##########################################

    # Via e-mail: crash when non-float32 compat
    d = dcrf.DenseCRF2D(10, 10, 2)
    d.setUnaryEnergy(np.ones((2, 10 * 10), dtype=np.float32))
    compat = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        d.addPairwiseBilateral(sxy=(3, 3), srgb=(3, 3, 3), rgbim=np.zeros(
            (10, 10, 3), np.uint8), compat=compat)
        d.inference(2)
