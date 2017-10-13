# distutils: language = c++
# distutils: sources = pydensecrf/densecrf/src/densecrf.cpp pydensecrf/densecrf/src/unary.cpp pydensecrf/densecrf/src/pairwise.cpp pydensecrf/densecrf/src/permutohedral.cpp pydensecrf/densecrf/src/optimization.cpp pydensecrf/densecrf/src/objective.cpp pydensecrf/densecrf/src/labelcompatibility.cpp pydensecrf/densecrf/src/util.cpp pydensecrf/densecrf/external/liblbfgs/lib/lbfgs.c
# distutils: include_dirs = pydensecrf/densecrf/include pydensecrf/densecrf/external/liblbfgs/include

from numbers import Number

import eigen
cimport eigen

import numpy as np
cimport numpy as np


cdef LabelCompatibility* _labelcomp(compat) except NULL:
    if isinstance(compat, Number):
        return new PottsCompatibility(compat)
    elif memoryview(compat).ndim == 1:
        return new DiagonalCompatibility(eigen.c_vectorXf(compat))
    elif memoryview(compat).ndim == 2:
        return new MatrixCompatibility(eigen.c_matrixXf(compat))
    else:
        raise ValueError("LabelCompatibility of dimension >2 not meaningful.")
    return NULL  # Important for the exception(s) to propagate!


cdef class PairwisePotentials:

    def __cinit__(
        self, 
        float[:,::1] features not None, 
        compat, KernelType kernel=DIAG_KERNEL,
        NormalizationType normalization=NORMALIZE_SYMMETRIC, *_, **__):
        # We need to swallow extra-arguments because superclass cinit function
        # will always be called with the same params as the subclass, automatically.

        # We also only want to avoid creating an object if we're just being called
        # from a subclass as part of the hierarchy.
        if type(self) is PairwisePotentials:
            self._this = new c_PairwisePotentials(
                eigen.c_matrixXf(features), _labelcomp(compat), ktype=kernel,
                ntype=normalization)
        else:
            self._this = NULL


    def apply(self, np.ndarray[float, ndim=2, mode="c"] inp not None):
        cdef MatrixXf in_matrix = eigen.matrixXf(inp)
        m, n = inp.shape[0], inp.shape[1]
        cdef MatrixXf out_matrix = eigen.matrixXf(np.zeros([m, n]).astype(np.float32))
        self._this.apply(out_matrix.m, in_matrix.m)
        return np.array(out_matrix)

    def __dealloc__(self):
        # Because destructors are virtual, this is enough to delete any object
        # of child classes too.
        if self._this:
            del self._this
