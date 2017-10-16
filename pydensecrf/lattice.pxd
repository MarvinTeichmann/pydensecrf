from eigen cimport *


cdef extern from "densecrf/include/labelcompatibility.h":
    cdef cppclass LabelCompatibility:
        pass

    cdef cppclass PottsCompatibility(LabelCompatibility):
        PottsCompatibility(float) except +
        void apply( c_MatrixXf & out_values, const c_MatrixXf & in_values ) const

    cdef cppclass DiagonalCompatibility(LabelCompatibility):
        DiagonalCompatibility(const c_VectorXf&) except +

    cdef cppclass MatrixCompatibility(LabelCompatibility):
        MatrixCompatibility(const c_MatrixXf&) except +

cdef extern from "densecrf/include/pairwise.h":
    cpdef enum NormalizationType: NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC
    cpdef enum KernelType: CONST_KERNEL, DIAG_KERNEL, FULL_KERNEL


cdef extern from "densecrf/include/permutohedral.h":
    cdef cppclass c_Permutohedral "Permutohedral":
        c_Permutohedral() except +
        void init ( const c_MatrixXf & features );
        c_MatrixXf compute ( const c_MatrixXf & v, bool) const;
        void compute ( c_MatrixXf & out, const c_MatrixXf &, bool ) const;
cdef class Permutohedral:
    cdef c_Permutohedral *_this