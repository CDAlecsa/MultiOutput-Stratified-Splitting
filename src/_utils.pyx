#cython: boundscheck = False
#cython: wraparound = False
#cython: cdivision = True

# Load modules
import numpy as np 

cimport cython

from cython.operator cimport dereference, preincrement

from libc.math cimport fabs, sqrt, pow
from libc.stdlib cimport qsort,  malloc, free
from libcpp.algorithm cimport find
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.map cimport map





cdef int compare_for_sort(const void* a, const void* b) noexcept nogil:
    '''
        Comparison function used for the `argsort` method.
    '''
    cdef:
        IndexValue struct_a = (<IndexValue*>a)[0]
        IndexValue struct_b = (<IndexValue*>b)[0]
    
    if struct_a.value < struct_b.value: return -1
    elif struct_a.value > struct_b.value: return 1
    else: return 0





cdef void argsort(const float64_t[:] x, float64_t[:] sorted_x, intp_t[:] sorted_indices) noexcept nogil:
    '''
        The actual argsort function.
    '''
    cdef:
        intp_t i, N
        IndexValue* sorted_array

    N = x.shape[0]
    sorted_array = <IndexValue*> malloc(N * sizeof(IndexValue))

    if sorted_array == NULL:
        raise MemoryError("Unable to allocate array")

    for i in range(N):
        sorted_array[i].index = i
        sorted_array[i].value = x[i]

    qsort(<void*> sorted_array, N, sizeof(IndexValue), compare_for_sort)

    for i in range(N):
        sorted_indices[i] = sorted_array[i].index
        sorted_x[i] = sorted_array[i].value

    free(sorted_array)





cdef intp_t[:] nonzero(const binary_int[:] x):
    '''
        A Cython implementation of `np.nonzero` for 1D boolean arrays
    '''
    cdef:
        intp_t i, N
        vector[intp_t] v
        intp_t[:] indices
    
    N = x.shape[0]
    for i in range(N):
        if x[i] != 0:
            v.push_back(i)
    
    indices = ( < intp_t [:v.size()] > v.data() ).copy()
    return indices





cdef float64_t norm_1d(const float64_t[:] x) noexcept nogil:
    '''
        A Cython implementation of `np.linalg.norm(..., ord = 1)` for 1D float arrays
    '''
    cdef:
        intp_t i, N
        float64_t out

    out = 0.0
    N = x.shape[0]

    for i in range(N):
        out = out + fabs(x[i])

    return out





cdef float64_t std(const float64_t[:] x) noexcept nogil:
    '''
        A Cython implementation of `np.std` for 1D float arrays
    '''
    cdef:
        intp_t i, N
        float64_t N_float, mean, out

    N = x.shape[0]
    N_float = <float64_t> N
    mean = 0.0
    out = 0.0

    for i in range(N):
        mean = mean + x[i]

    mean = mean / N_float

    for i in range(N):
        out += pow(x[i] - mean, 2) 

    out = out / N_float
    out = sqrt(out)
    return out





cdef integer[:] hstack(const integer[:] x, const integer[:] y):
    '''
        A Cython implementation of `np.hstack` for 1D boolean or int arrays
    '''
    cdef:
        intp_t i, N_x, N_y
        vector[integer] v
        integer[:] out
        

    N_x = x.shape[0]
    N_y = y.shape[0]

    for i in range(N_x):
        v.push_back(x[i])

    for i in range(N_y):
        v.push_back(y[i])  

    out = ( < integer [:v.size()] > v.data() ).copy()
    return out
    




cdef integer[:, :] vstack(const integer[:, :] x,
                          const integer[:, :] y,
                          const integer[:, :] z):
    '''
        A Cython implementation of `np.vstack` for 2D boolean or int arrays
    '''
    cdef:
        intp_t i, count, N_x, N_y, N_z, N_total, cols
        integer[:, :] out
        type np_dtype
        
        
    N_x = x.shape[0]
    N_y = y.shape[0]
    N_z = z.shape[0]
    cols = x.shape[1]
    N_total = N_x + N_y + N_z

    if integer is binary_int:
        np_dtype = np.uint8
    elif integer is intp_t:
        np_dtype = np.intp


    out = np.empty(shape = (N_total, cols), dtype = np_dtype)
    count = 0

    for i in range(N_x):
        out[count, :] = x[i, :]
        count = count + 1

    for i in range(N_y):
        out[count, :] = y[i, :]
        count = count + 1

    for i in range(N_z):
        out[count, :] = z[i, :]
        count = count + 1

    return out





cdef intp_t[:, :] unique_counts(const intp_t[:] x):
    '''
        A Cython implementation of `np.unique(..., return_counts = True)` for 1D int arrays
    '''
    cdef:
        intp_t i, N, elem
        map[intp_t, intp_t] counts
        map[intp_t, intp_t].const_iterator c_iterator
        set[intp_t] s
        intp_t[:, :] out


    N = x.shape[0]

    for i in range(N):
        elem = x[i]
        if (s.insert(elem).second):
            counts[elem] = 1
        else:
            counts[elem] += 1    

    c_iterator = counts.begin()
    out = np.empty(shape = (counts.size(), 2), dtype = np.intp)

    i = 0
    while c_iterator != counts.end() :
        out[i, 0] = dereference(c_iterator).first
        out[i, 1] = dereference(c_iterator).second
        i = i + 1
        preincrement(c_iterator)

    return out
    




cdef intp_t[:] union_1d(const intp_t[:] x, const intp_t[:] y):
    '''
        A Cython implementation of `np.union1d` for 1D int arrays
    '''
    cdef:
        intp_t[:] stacked_array, out
        
    stacked_array = hstack[intp_t](x, y)
    out = unique_counts(stacked_array)[:, 0]
    return out





cdef binary_int[:] isin_1d(const intp_t[:] x, const intp_t[:] y):
    '''
        A Cython implementation of `np.isin` for 1D int arrays
    '''
    cdef:
        intp_t i, N_x, N_y
        vector[intp_t] y_vect
        vector[intp_t].iterator iterator
        binary_int[:] out


    N_x = x.shape[0]
    N_y = y.shape[0]
    out = np.zeros(shape = (N_x, ), dtype = np.uint8)


    for i in range(N_y):
        y_vect.push_back(y[i])

    
    for i in range(N_x):
        iterator = find(y_vect.begin(), y_vect.end(), x[i])
        if ( iterator != y_vect.end() ):
            out[i] = 1

    return out





cdef binary_int[:, :] take_2d(binary_int[:, :] x, intp_t[:] indices):
    '''
        A Cython implementation of `np.take` for 2D boolean arrays
    '''
    cdef:
        intp_t i, N, cols
        binary_int[:, :] out

    N = indices.shape[0]
    cols = x.shape[1]

    out = np.empty(shape = (N, cols), dtype = np.uint8)

    for i in range(N):    
        out[i, :] = x[ indices[i], : ]

    return out

