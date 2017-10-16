# distutils: language = c++
# distutils: sources = pydensecrf/densecrf/src/densecrf.cpp pydensecrf/densecrf/src/permutohedral.cpp pydensecrf/densecrf/src/unary.cpp pydensecrf/densecrf/src/pairwise.cpp pydensecrf/densecrf/src/permutohedral.cpp pydensecrf/densecrf/src/optimization.cpp pydensecrf/densecrf/src/objective.cpp pydensecrf/densecrf/src/labelcompatibility.cpp pydensecrf/densecrf/src/util.cpp pydensecrf/densecrf/external/liblbfgs/lib/lbfgs.c
# distutils: include_dirs = pydensecrf/densecrf/include pydensecrf/densecrf/external/liblbfgs/include

from numbers import Number

import eigen
cimport eigen

import numpy as np
cimport numpy as np


cdef class Permutohedral:

    def __cinit__(self):
        self._this = new c_Permutohedral()

    def init_filer(self, float[:,::1] features not None):
        self._this.init(eigen.c_matrixXf(features))


    def compute (self, np.ndarray[float, ndim=2, mode="c"] inp not None, bint reverse=False ):

        cdef MatrixXf in_matrix = eigen.matrixXf(inp)
        m, n = inp.shape[0], inp.shape[1]
        cdef MatrixXf out_matrix = eigen.matrixXf(np.zeros([m, n]).astype(np.float32))
        self._this.compute(out_matrix.m, in_matrix.m, reverse)
        return np.array(out_matrix)
