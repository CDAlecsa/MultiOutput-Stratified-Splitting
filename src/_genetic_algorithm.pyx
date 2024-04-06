#cython: boundscheck = False
#cython: wraparound = False
#cython: cdivision = True

# Load modules
import numpy as np

from _utils cimport (argsort, 
                     nonzero, 
                     norm_1d, 
                     std, 
                     hstack, 
                     vstack,
                     unique_counts, 
                     union_1d, 
                     isin_1d,
                     take_2d)
                     
from libc.stdio cimport printf



# Numpy types
from numpy import intp as INTP
from numpy import float32 as FLOAT32
from numpy import float64 as FLOAT64
from numpy import uint8 as UINT8







########### Genetic algorithm for train-test split ###########
cdef class GeneticAlgorithm:


    def __cinit__(
                self,
                n_iterations,
                n_samples,
                n_outputs,
                N_train, 
                N_test,
                population_size,
                mutation_size,
                crossover_size,
                n_individuals_by_mutation,
                n_individuals_by_crossover,
                sample_with_replacement,
                verbose,
                random_state
                ):

        self.n_iterations = n_iterations
        
        self.n_samples = n_samples
        self.n_outputs = n_outputs

        self.N_train = N_train
        self.N_test = N_test

        self.population_size = population_size
        self.mutation_size = mutation_size
        self.crossover_size = crossover_size

        self.n_individuals_by_mutation = n_individuals_by_mutation
        self.n_individuals_by_crossover = n_individuals_by_crossover

        self.sample_with_replacement = sample_with_replacement
        self.verbose = verbose
        self.random_state = random_state





    cdef binary_int[:, :] init_population(self):
        '''
            Create the individuals of the population by using a randomized train/test split.
        '''
        cdef:
            intp_t i, population_size_
            binary_int[:] generic_individual
            binary_int[:, :] population

        population_size_ = self.population_size
        population = np.empty( shape = (population_size_, self.n_samples), dtype = UINT8 )

        # [train = 0] & [test = 1]
        generic_individual = hstack[binary_int]( np.zeros(shape = (self.N_train, ), dtype = UINT8), 
                                                 np.ones(shape = (self.N_test, ), dtype = UINT8) 
                                            )
        
        for i in range(population_size_):
            population.base[i, :] = self.random_state.permutation(generic_individual)

        return population





    cpdef list fit(self, const intp_t[:, ::1] y):
        '''
            Training of the genetic algorithm.
        '''
        cdef:
            intp_t i, j, k, mutation_idx
            intp_t n_iterations_, population_size_
            intp_t n_samples_, N_train_, N_test_, 
            intp_t n_individuals_by_mutation_, n_individuals_by_crossover_

            intp_t[:] crossover_idx, best_ind_train, best_ind_test
            intp_t[:] population_indices_by_fitness, stacked_population_indices_by_fitness
            
            intp_t[:, ::1] train_indices, test_indices 

            float64_t best_fitness
            float64_t[:] population_fitnesses, stacked_population_fitnesses
            float64_t[::1] losses, stds

            binary_int[:] best, not_best
            binary_int[:, :] population, stacked_population, mutation_childrens, crossover_childrens

            list results



        # Initialize the multioutput target
        self.y = y

        # Define local variables
        n_iterations_ = self.n_iterations
        n_samples_ = self.n_samples
        N_train_ = self.N_train
        N_test_ = self.N_test
        population_size_ = self.population_size
        n_individuals_by_mutation_ = self.n_individuals_by_mutation
        n_individuals_by_crossover_ = self.n_individuals_by_crossover

        # Initialize outputs
        losses = np.empty( shape = (n_iterations_, ) ).astype(FLOAT64)
        stds = np.empty( shape = (n_iterations_, ) ).astype(FLOAT64)

        best_ind_train = np.empty( shape = (N_train_, ) ).astype(INTP)
        best_ind_test = np.empty( shape = (N_test_, ) ).astype(INTP)

        train_indices = np.empty( shape = (n_iterations_, N_train_) ).astype(INTP)
        test_indices = np.empty( shape = (n_iterations_, N_test_) ).astype(INTP)

        # Initialize variables for the genetic process
        crossover_childrens = np.empty( shape = (n_individuals_by_crossover_, n_samples_) ).astype(UINT8)
        mutation_childrens = np.empty( shape = (n_individuals_by_mutation_, n_samples_) ).astype(UINT8)

        # Initialize variable corresponding to the stacked population
        stacked_population = np.empty( shape = (population_size_ + n_individuals_by_crossover_ + n_individuals_by_mutation_, 
                                                n_samples_) ).astype(UINT8)

        # Initialize population
        population = self.init_population()




        # Main loop
        for i in range(n_iterations_):

            # Apply crossover
            for k in range(0, n_individuals_by_crossover_, 2):
                crossover_idx = self.random_state.choice(population_size_, size = 2, replace = self.sample_with_replacement).astype(INTP)
                crossover_childrens.base[k : k + 2, :] = self.crossover( population.base[crossover_idx] )


            # Apply mutation
            for k in range(n_individuals_by_mutation_):
                mutation_idx = self.random_state.choice(population_size_, size = 1, replace = self.sample_with_replacement)[0]
                mutation_childrens.base[k, :] = self.mutation( population[mutation_idx] )  
            

            # Apply selection
            stacked_population = vstack[binary_int]( population, crossover_childrens, mutation_childrens )

            stacked_population_indices_by_fitness = np.empty(shape = (stacked_population.shape[0], ), dtype = INTP)
            stacked_population_fitnesses = np.empty(shape = (stacked_population.shape[0], ), dtype = FLOAT64)
            self.rank_population(stacked_population, stacked_population_fitnesses, stacked_population_indices_by_fitness)

            population_indices_by_fitness = stacked_population_indices_by_fitness.base[: population_size_]
            population_fitnesses = stacked_population_fitnesses.base[: population_size_]

            population = take_2d(stacked_population, population_indices_by_fitness)


            # Get best individual
            best = population[0].copy()
            best_fitness = population_fitnesses[0]

            # Store the best fitness value & standard deviation of the fitnesses at the current iteration 
            losses[i] = best_fitness
            stds[i] = std( population_fitnesses )

            # Printing options
            if self.verbose:
                printf( "Iteration: [%zd]/[%zd] \t Loss: [%1.3f] \t Std: [%1.3f]\n", i, n_iterations_ - 1, losses[i], stds[i] )


            # Get the best indices for the current iteration
            # the test indices of `best` are marked with 1
            # the train indices of `not_best` are marked with 1
            not_best = np.empty(shape = (n_samples_, ), dtype = UINT8)

            for j in range(n_samples_):            
                not_best[j] = 1 - best[j]

            best_ind_train = nonzero( not_best )
            best_ind_test = nonzero( best )

            train_indices[i] = best_ind_train
            test_indices[i] = best_ind_test



        results = [losses, stds, best_ind_train, best_ind_test, train_indices, test_indices]
        return results
        




    cdef float64_t objective(self, binary_int[:] ind):
        '''
            The fitness function.
        '''
        cdef:
            intp_t i, k
            intp_t i_train_count, i_test_count
            intp_t N_samples, N_train, N_test, n_outputs
            intp_t N_train_unique_values, N_test_unique_values, N_all_unique_values
            
            intp_t[:] ind_train, ind_test
            intp_t[:] train_values, test_values, all_values
            intp_t[:] train_counts, test_counts, total_counts

            intp_t[:, :] y_train, y_test
            intp_t[:, :] train_uc, test_uc

            float64_t loss_ = 0.0, eps = 1e-10
            
            float64_t[:] train_perc, test_perc
            float64_t[:] all_train_perc, all_test_perc, all_total_perc
            float64_t[:] diff_perc_train_total, diff_perc_test_total

            binary_int[:] individual, not_individual
            binary_int[:] train_idx, test_idx



        # Define variables
        N_samples = self.n_samples       
        n_outputs = self.n_outputs

        # Define the individual which aids in the computation of the train/test indices
        individual = ind.copy()

        # the test indices of `individual` are marked with 1
        # the train indices of `not_individual` are marked with 1
        not_individual = np.empty(shape = (N_samples, ), dtype = UINT8)

        for i in range(N_samples):            
            not_individual[i] = 1 - individual[i]

        ind_train = nonzero( not_individual )
        ind_test = nonzero( individual )

        # Store the train & test components corresponding to the target
        y_train = self.y.base[ind_train]
        y_test = self.y.base[ind_test]

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]



        # Loop over the number of target outputs        
        for k in range(n_outputs):

            # Determine the unique values & counts with respect to the train subset
            train_uc = unique_counts(y_train[:, k])
            train_values = train_uc[:, 0]
            train_counts = train_uc[:, 1]

            # Determine the unique values & counts with respect to the test subset
            test_uc = unique_counts(y_test[:, k])
            test_values = test_uc[:, 0]
            test_counts = test_uc[:, 1]

            # Store the number of unique values from the train/test subsets
            N_train_unique_values = train_values.shape[0]
            N_test_unique_values = test_values.shape[0]

            # Compute the percentages with respect to the train/test counts
            train_perc = np.empty(shape = (N_train_unique_values, ), dtype = FLOAT64)
            test_perc = np.empty(shape = (N_test_unique_values, ), dtype = FLOAT64)

            for i in range(N_train_unique_values):
                train_perc[i] = train_counts[i] / ( N_train - train_counts[i] + eps )

            for i in range(N_test_unique_values):
                test_perc[i] = test_counts[i] / ( N_test - test_counts[i] + eps )


            # Determine all the unique values
            all_values = union_1d(train_values, test_values)

            # Define the memoryviews in which we store the percentages with respect to 
            # the counts corresponding to all the unique values
            all_train_perc = np.zeros_like(all_values, dtype = FLOAT64)
            all_test_perc = np.zeros_like(all_values, dtype = FLOAT64)

            # Compute the binary indices corresponding to the values that belong to the train/test subsets
            train_idx = isin_1d(all_values, train_values)
            test_idx = isin_1d(all_values, test_values)

            # Compute the counts percentages by taking into account all the unique values
            N_all_unique_values = all_values.shape[0]

            i_train_count = 0
            i_test_count = 0

            '''

                * [Some observations regarding the values returned by different custom Cython functions]:
                    - The `train_values` & `test_values` memoryviews contain the unique values of train/test subsets, and are <sorted>.
                    - The `train_counts` & `test_counts` memoryviews are also <sorted> since, for a fixed index, 
                        each count corresponds to a specific unique value.
                    - The `union_1d` function returns a <sorted> memoryview containing the values present in both train & test subsets.
                    - The `isin_1d` function returns boolean values which matches 
                        the values from `all_values` that belong to `train_values`/`test_values`, 
                        hence the boolean values appear in the same order as the <sorted> values from `all_values`.

                * [Here we motivate the fact that the computations made in the `for loop` given below are well defined]:
                    - Since `all_values` is <sorted>, and `train_perc` is computed using `train_counts`
                        (where the latter one is <sorted> with respect to `train_values`),
                        then each time we find an index for which the value from `all_values` belongs to `train_values` 
                        we shall retrieve the corresponding value from `train_perc` in a <sorted> manner.
                    - Consequently, in this iterative process, for a given `all_values` index,
                        we are guaranteed to use the correct value of `train_perc` due to the aforementioned sorting.
                    - The case of `all_test_perc` is similar.
            '''

            for i in range(N_all_unique_values):

                if train_idx[i] == 1:
                    all_train_perc[i] = train_perc[i_train_count]
                    i_train_count = i_train_count + 1

                if test_idx[i] == 1:
                    all_test_perc[i] = test_perc[i_test_count]
                    i_test_count = i_test_count + 1




            # Determine the unique values & counts with respect to the whole set
            total_counts = unique_counts(self.y.base[:, k])[:, 1]
            
            # Compute the percentages with respect to the counts
            all_total_perc = np.empty(shape = (N_all_unique_values, ), dtype = FLOAT64)

            for i in range(N_all_unique_values):
                all_total_perc[i] = total_counts[i] / ( N_samples - total_counts[i] + eps )




            # Compute the loss value corresponding to the current target output
            diff_perc_train_total = np.zeros(shape = (N_all_unique_values, ), dtype = FLOAT64)
            diff_perc_test_total = np.zeros(shape = (N_all_unique_values, ), dtype = FLOAT64)

            for i in range(N_all_unique_values):
                diff_perc_train_total[i] = all_train_perc[i] - all_total_perc[i]
                diff_perc_test_total[i] = all_test_perc[i] - all_total_perc[i]

            loss_ = loss_ + 0.5 * ( norm_1d(diff_perc_train_total) + norm_1d(diff_perc_test_total) )


        loss_ = loss_ / (<float64_t> n_outputs)
        return loss_




    
    cdef binary_int[:] mutation(self, const binary_int[:] ind):
        '''
            Mutation process: A modified variant of bit-flip mutation.
        '''
        cdef:
            intp_t i, N_samples, N_mutation
            binary_int[:] individual, not_individual
            intp_t[:] ind_train, ind_test, ind_train_to_swap, ind_test_to_swap



        # Define variables
        N_samples = self.n_samples
        N_mutation = self.mutation_size

        # Define the individual which will be mutated
        individual = ind.copy()

        # the test indices of `individual` are marked with 1
        # the train indices of `not_individual` are marked with 1
        not_individual = np.empty(shape = (N_samples, ), dtype = UINT8)

        for i in range(N_samples):            
            not_individual[i] = 1 - individual[i]

        # Determine the indices corresponding to train/test subsets
        ind_train = nonzero( not_individual )
        ind_test = nonzero( individual )
        
        # Select randomly train/test indices
        ind_train_to_swap = self.random_state.permutation( ind_train )[ : N_mutation ]
        ind_test_to_swap = self.random_state.permutation( ind_test )[ : N_mutation ]

        # Apply mutation by swapping train & test indices
        for i in range(N_mutation):
            individual[ind_train_to_swap[i]] = 1              
            individual[ind_test_to_swap[i]] = 0  

        return individual





    cdef binary_int[:, :] crossover(self, const binary_int[:, :] ind):
        '''
            Crossover process: Swap the train & test indices between two individuals.
        '''
        cdef:
            binary_int[:, :] individual_out


        # Apply the crossover between the 2 parents
        individual_out = self._basic_crossover(ind)

        # Reassign indices to preserve the train/test ratio
        individual_out.base[0, :] = self._reassign(individual_out.base[0, :])
        individual_out.base[1, :] = self._reassign(individual_out.base[1, :])

        return individual_out





    cdef binary_int[:, :] _basic_crossover(self, const binary_int[:, :] ind):
        '''
            Crossover process: Swap the train & test indices between two individuals.
        '''
        cdef:
            intp_t i, N_samples, N_crossover
            intp_t[:] ind_1_to_swap, ind_2_to_swap

            binary_int[:, :] individual_out

            binary_int[:] individual_1, individual_2
            binary_int[:] elements_from_1_to_2, elements_from_2_to_1


        # Define variables
        N_samples = self.n_samples
        N_crossover = self.crossover_size

        # Define the individuals for which we will apply the crossover process
        individual_1 = ind[0, :].copy()
        individual_2 = ind[1, :].copy()

        # Define memoryviews
        individual_out = np.empty(shape = (2, N_samples), dtype = UINT8)
        elements_from_1_to_2 = np.empty(shape = (N_crossover, ), dtype = UINT8)
        elements_from_2_to_1 = np.empty(shape = (N_crossover, ), dtype = UINT8)

        # Select randomly the indices to swap between the 2 individuals
        ind_1_to_swap = self.random_state.permutation( np.arange(N_samples, dtype = INTP) )[ : N_crossover ]
        ind_2_to_swap = self.random_state.permutation( np.arange(N_samples, dtype = INTP) )[ : N_crossover ]
        
        # Retain the train/test elements to swap between the 2 individuals
        for i in range(N_crossover):
            elements_from_1_to_2[i] = individual_1[ind_1_to_swap[i]]
            elements_from_2_to_1[i] = individual_2[ind_2_to_swap[i]]

        # Swap the elements between the 2 individuals
        for i in range(N_crossover):
            individual_1[ind_1_to_swap[i]] = elements_from_2_to_1[i]
            individual_2[ind_2_to_swap[i]] = elements_from_1_to_2[i]

        # Put together the new individuals
        individual_out.base[0, :] = individual_1
        individual_out.base[1, :] = individual_2

        return individual_out





    cdef binary_int[:] _reassign(self, const binary_int[:] ind):
        '''
            Reassign the indices in order to keep the original train/test ratio.
        '''
        cdef:
            intp_t i, N_samples, n_train_, n_test_
            intp_t n_train_to_swap, n_test_to_swap

            intp_t[:] ind_train, ind_test, ind_train_to_swap, ind_test_to_swap
            binary_int[:] individual, not_individual


        # Define the number of samples
        N_samples = self.n_samples

        # Define the individual for which we apply the reassign process
        individual = ind.copy()

        # the test indices of `individual` are marked with 1
        # the train indices of `not_individual` are marked with 1
        not_individual = np.empty(shape = (N_samples, ), dtype = UINT8)

        for i in range(N_samples):            
            not_individual[i] = 1 - individual[i]
        
        # Determine the indices corresponding to train/test subsets
        ind_train = nonzero( not_individual )
        ind_test = nonzero( individual )

        # Determine the number of the train/test indices obtained after the crossover process
        n_train_ = ind_train.shape[0]
        n_test_ = ind_test.shape[0]

        # Determine the number of train/test indices which needs to be swapped
        # `n_train_to_swap` is equal to `n_test_to_swap`
        n_train_to_swap = self.N_train - n_train_
        n_test_to_swap = self.N_test - n_test_

        # The reassign process in which there are missing a certain number of train elements
        if n_train_to_swap > 0: 
            ind_test_to_swap = self.random_state.permutation(ind_test)[ : n_train_to_swap ]
            for i in range(n_train_to_swap):
                individual[ind_test_to_swap[i]] = 0

        # The reassign process in which there are missing a certain number of test elements
        elif n_test_to_swap > 0:
            ind_train_to_swap = self.random_state.permutation(ind_train)[ : n_test_to_swap ]
            for i in range(n_test_to_swap):
                individual[ind_train_to_swap[i]] = 1

        return individual





    cdef void rank_population(self, binary_int[:, :] population, 
                                    float64_t[:] sorted_fitnesses, 
                                    intp_t[:] sorted_indices
                            ):
        '''
            Rank the population according to the fitness function.
        '''
        cdef:
            intp_t i, N
            binary_int[:] individual
            float64_t[:] losses

        # Define variables
        N = population.shape[0]
        losses = np.empty(shape = (N, ), dtype = FLOAT64)

        # Compute fitness values
        for i in range(N):
            individual = population[i]
            losses[i] = self.objective(individual)

        # Sort the fitness values
        argsort(losses, sorted_fitnesses, sorted_indices)
        
        
