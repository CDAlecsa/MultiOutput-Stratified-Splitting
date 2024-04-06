# distutils: language = c++

# Load modules
import numpy as np
cimport numpy as cnp





# Types
ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t
ctypedef cnp.uint8_t binary_int



# Custom fused type for integers
ctypedef fused integer:
    binary_int
    intp_t


# Custom structure related to the argsort functions
cdef struct IndexValue:
    float64_t value
    intp_t index







# Functions
cdef int compare_for_sort(const void* a, const void* b) noexcept nogil

cdef void argsort(const float64_t[:] x, float64_t[:] sorted_x, intp_t[:] sorted_indices) noexcept nogil

cdef intp_t[:] nonzero(const binary_int[:] x)

cdef float64_t norm_1d(const float64_t[:] x) noexcept nogil

cdef float64_t std(const float64_t[:] x) noexcept nogil

cdef integer[:] hstack(const integer[:] x, const integer[:] y)

cdef integer[:, :] vstack(const integer[:, :] x,
                          const integer[:, :] y,
                          const integer[:, :] z)

cdef intp_t[:, :] unique_counts(const intp_t[:] x)

cdef intp_t[:] union_1d(const intp_t[:] x, const intp_t[:] y)

cdef binary_int[:] isin_1d(const intp_t[:] x, const intp_t[:] y)

cdef binary_int[:, :] take_2d(binary_int[:, :] x, intp_t[:] indices)

