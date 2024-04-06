# distutils: language = c++

# Load modules
cimport numpy as cnp
cnp.import_array()

from _utils cimport (binary_int, 
                     intp_t, 
                     float32_t, 
                     float64_t)







# Genetic algorithm for train-test split
cdef class GeneticAlgorithm:
    

    # Variables
    cdef intp_t n_iterations
    cdef intp_t n_samples
    cdef intp_t n_outputs
    cdef intp_t population_size
    cdef intp_t n_individuals_by_mutation
    cdef intp_t n_individuals_by_crossover

    cdef intp_t N_test
    cdef intp_t N_train
    cdef intp_t mutation_size
    cdef intp_t crossover_size

    cdef bint sample_with_replacement
    cdef bint verbose
    cdef object random_state

    cdef const intp_t[:, ::1] y





    # Methods
    cdef binary_int[:, :] init_population(self)

    cdef binary_int[:] _reassign(self, const binary_int[:] ind)
    cdef binary_int[:, :] _basic_crossover(self, const binary_int[:, :] ind)
    cdef binary_int[:, :] crossover(self, const binary_int[:, :] ind)

    cdef binary_int[:] mutation(self, const binary_int[:] ind)

    cdef void rank_population(self, binary_int[:, :] population, 
                                    float64_t[:] sorted_fitnesses, 
                                    intp_t[:] sorted_population_indices
                            )
                            
    cdef float64_t objective(self, binary_int[:] ind)

    cpdef list fit(self, const intp_t[:, ::1] y)

