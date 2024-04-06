# Load modules
import itertools

import numpy as np
import numpy.typing as npt

from misc import timeit, make_counts
from _genetic_algorithm import INTP, FLOAT32, FLOAT64

from typing import Union
from numbers import Integral

from sklearn.utils.multiclass import type_of_target
from sklearn.utils import (check_array, 
                           check_random_state)
from sklearn.utils._param_validation import (Interval, 
                                             RealNotInt, 
                                             validate_parameter_constraints)

from _genetic_algorithm import GeneticAlgorithm







########### Splitter class for multioutput targets ###########
class MultiOutputStratifiedSplitter():


    # Constraints
    _parameter_constraints: dict = {
        "n_iterations": [Interval(Integral, 1, None, closed = "left")],
        "test_size": [Interval(RealNotInt, 0.0, 1.0, closed = "neither")],
        "population_size": [Interval(Integral, 2, None, closed = "left")],
        "mutation_rate": [
                            Interval(RealNotInt, 0.0, 1.0, closed = "neither"),
                            Interval(Integral, 1, None, closed = "left"),
                          ],
        "crossover_rate": [
                            Interval(RealNotInt, 0.0, 1.0, closed = "neither"),
                            Interval(Integral, 1, None, closed = "left"),
                            ],
        "n_individuals_by_mutation": [Interval(Integral, 1, None, closed = "left")],
        "n_individuals_by_crossover": [Interval(Integral, 1, None, closed = "left")],
        "sample_with_replacement": [bool],
        "verbose": ["verbose"],
        "random_state": ["random_state"]
    }




    def __init__(self, 
                 n_iterations: int = 100,
                 test_size: float = 0.2,
                 population_size: int = 100,
                 mutation_rate: Union[float, int] = 0.25,
                 crossover_rate: Union[float, int] = 1,
                 n_individuals_by_mutation: int = 20,
                 n_individuals_by_crossover: int = 20,
                 sample_with_replacement: bool = True,
                 verbose: bool = False,
                 random_state: Union[None, int, np.random.RandomState] = None):

        self.n_iterations = n_iterations
        self.test_size = test_size
        self.population_size = population_size

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.n_individuals_by_crossover = n_individuals_by_crossover
        self.n_individuals_by_mutation = n_individuals_by_mutation

        self.sample_with_replacement = sample_with_replacement
        self.verbose = verbose
        self.random_state = random_state
        self.check_params()

        self.train_size = None
        self.N_train = None
        self.N_test = None
        self.mutation_size = None
        self.crossover_size = None
        
        self.n_samples = None
        self.n_outputs = None

        self.losses = None
        self.stds = None

        self.train_idx = None
        self.test_idx = None

        self.train_indices = None
        self.test_indices = None





    def get_params(self):
        '''
            Retrieves a dictionary corresponding to the class' parameters.
        '''
        _params = { k: v for (k, v) in zip( self._parameter_constraints.keys(), 
                                           [self.n_iterations,
                                            self.test_size,
                                            self.population_size,
                                            self.mutation_rate, 
                                            self.crossover_rate, 
                                            self.n_individuals_by_mutation,
                                            self.n_individuals_by_crossover,
                                            self.sample_with_replacement,
                                            self.verbose,
                                            self.random_state] ) }
        return _params
    


    

    def check_params(self):
        '''
            Check the initialization values for errors.
        '''

        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(),
            caller_name = self.__class__.__name__
        )


    


    def check_data(self, X: npt.NDArray[FLOAT32], y: npt.NDArray[INTP]):
        '''
            Check if the data pair (X, y) is well defined.
        '''

        X = check_array(X, input_name = "X", accept_sparse = True, allow_nd = True, dtype = FLOAT32)
        y = check_array(y, input_name = "y", accept_sparse = False, ensure_2d = True, dtype = INTP, 
                                                                                      order = 'C')

        type_of_target_ = type_of_target(y, input_name = "y")
        allowed_target_type = "multiclass-multioutput"
        if type_of_target_ != allowed_target_type:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_type, type_of_target_
                )
            )

        return X, y 
        




    def init_params(self):
        '''
            Initialize parameters which will be used by the genetic algorithm.
        '''
        self.N_test = int( self.test_size * self.n_samples )
        self.N_train = self.n_samples - self.N_test
        self.train_size = self.N_train / self.n_samples


        if self.N_train == 0:
            raise ValueError(
                "With n_samples={}, test_size={} and train_size={}, the "
                "resulting `train set` will be empty. Adjust any of the "
                "aforementioned parameters.".format(self.n_samples, self.test_size, self.train_size)
            )
        if self.N_test == 0:
            raise ValueError(
                "With n_samples={}, test_size={} and train_size={}, the "
                "resulting `test set` will be empty. Adjust any of the "
                "aforementioned parameters.".format(self.n_samples, self.test_size, self.train_size)
            )



        # The mutation size must be at most `min(N_test, N_train)`
        mutation_bound = min(self.N_test, self.N_train)

        if isinstance(self.mutation_rate, Integral):
            if self.mutation_rate > mutation_bound:
                msg = "`mutation_rate` must be <= min({},{}) but got value {}."
                raise ValueError(msg.format(self.N_test, self.N_train, self.mutation_rate))
            self.mutation_size = self.mutation_rate

        elif isinstance(self.mutation_rate, RealNotInt):
            self.mutation_size = int(self.mutation_rate * mutation_bound)




        # We use the same crossover_rate for all the samples indices
        if isinstance(self.crossover_rate, Integral):
            if self.crossover_rate > self.n_samples:
                msg = "`crossover_rate` must be <= n_samples={} but got value {}."
                raise ValueError(msg.format(self.n_samples, self.crossover_rate))
            self.crossover_size = self.crossover_rate

        elif isinstance(self.crossover_rate, RealNotInt):
            self.crossover_size = int( self.crossover_rate * self.n_samples )





    @timeit
    def fit(self, X: npt.NDArray[FLOAT32], y: npt.NDArray[INTP]):
        '''
            Fit the underlying genetic algorithm on the data pair (X, y).
        '''
        X, y = self.check_data(X, y)
        self.n_samples, self.n_outputs = y.shape

        self.init_params()

        # Fit the genetic algorithm
        rng = check_random_state(self.random_state)
        GA = GeneticAlgorithm(n_iterations = self.n_iterations,
                              n_samples = self.n_samples,
                              n_outputs = self.n_outputs,
                              N_train = self.N_train,
                              N_test = self.N_test,
                              population_size = self.population_size,
                              mutation_size = self.mutation_size, 
                              crossover_size = self.crossover_size,
                              n_individuals_by_mutation = self.n_individuals_by_mutation,
                              n_individuals_by_crossover = self.n_individuals_by_crossover,
                              sample_with_replacement = self.sample_with_replacement,
                              verbose = self.verbose,
                              random_state = rng
                            )

        [self.losses, 
         self.stds, 
         self.train_idx, 
         self.test_idx, 
         self.train_indices, 
         self.test_indices] = GA.fit(y)
        
        # Check if the train/test ratio is preserved after training the genetic algorithm
        assert ( self.N_train == len(self.train_idx) and self.N_test == len(self.test_idx) )
        del GA

        self.losses = np.array(self.losses, dtype = FLOAT64)
        self.stds = np.array(self.stds, dtype = FLOAT64)
        self.train_idx = np.array(self.train_idx, dtype = INTP)
        self.test_idx = np.array(self.test_idx, dtype = INTP)
        self.train_indices = np.array(self.train_indices, dtype = INTP)
        self.test_indices = np.array(self.test_indices, dtype = INTP)
        
        # Return the (X, y) pair corresponding to train/test subsets
        X_train = X[self.train_idx]
        X_test = X[self.test_idx]

        y_train = y[self.train_idx]
        y_test = y[self.test_idx]

        return X_train, X_test, y_train, y_test







def grid_search_param(X: npt.NDArray[FLOAT32],
                      y: npt.NDArray[INTP],
                      random_state: Union[None, int, np.random.RandomState],
                      test_size: Union[float, list],
                      n_iterations: Union[int, list],
                      population_size: Union[int, list],
                      mutation_rate: Union[float, int, list],
                      crossover_rate: Union[float, int, list],
                      n_individuals_by_mutation: Union[int, list],
                      n_individuals_by_crossover: Union[int, list],
                      sample_with_replacement: bool
                    ):
    '''
        Make a grid search over the genetic algorithm parameters.
    '''

    # Don't display the iterations of every splitter
    verbose = False


    # Define the dictionary corresponding to the hyper-parameters
    param_dict = {
        'test_size': test_size,
        'n_iterations': n_iterations,
        'population_size': population_size,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'n_individuals_by_mutation': n_individuals_by_mutation,
        'n_individuals_by_crossover': n_individuals_by_crossover,
        'sample_with_replacement': sample_with_replacement
    }
    param_names = list(param_dict.keys())
    

    # Rewrite dictionary values
    for k in param_names:
    
        if isinstance(param_dict[k], list):

            if len(param_dict[k]) != 3:
                raise ValueError("For a hyper-parameter, a list must be given containing exactly 3 elements.")

            if all([ isinstance( i, int ) for i in param_dict[k] ]):
                param_dict[k] = np.arange(*param_dict[k])
            elif all([ isinstance( param_dict[k][i], float ) for i in range(2) ]) and isinstance(param_dict[k][2], int):
                param_dict[k] = map(lambda x: round(x, 2), np.linspace(*param_dict[k]))
            else:
                raise ValueError("For a hyper-parameter, a homogeneous list must be given containing only `int` or `float` elements.")

        else:
            param_dict[k] = [param_dict[k]]


    # Define the combinations of hyper-parameters
    param_combinations = list(itertools.product( *param_dict.values() ))
    param_combinations = [ { param_names[c] : v for (c, v) in enumerate(i) } for i in param_combinations ]
    n_combinations = len(param_combinations)
    del param_dict, param_names


    
    # Loop over the parameters
    best_result = np.inf
    best_params = None
    best_splitter = None

    for count, p in enumerate(param_combinations):
        print('ITER: ', count + 1, '/', n_combinations, '\n', p)

        # Define & fit the current splitter object
        stratified_splitter = MultiOutputStratifiedSplitter(n_iterations = p["n_iterations"],
                                                            test_size = p["test_size"],
                                                            population_size = p["population_size"],
                                                            mutation_rate = p["mutation_rate"], 
                                                            crossover_rate = p["crossover_rate"],
                                                            n_individuals_by_mutation = p["n_individuals_by_mutation"],
                                                            n_individuals_by_crossover = p["n_individuals_by_crossover"],
                                                            sample_with_replacement = p["sample_with_replacement"],
                                                            verbose = verbose,
                                                            random_state = random_state)

        _ = stratified_splitter.fit(X, y)
        train_idx, test_idx = stratified_splitter.train_idx, stratified_splitter.test_idx

        n_outputs = y.shape[1]
        indices = list(itertools.combinations(range(n_outputs), r = 2))


        # Empty lists in which we will store the results for the current splitter
        diff_values_medians, diff_values_stds, diff_result = [], [], None


        # Gather statistics for the current tuple of target outputs
        for i in range(len(indices)):
            output_idx = indices[i]

            # Check the number of unique values per target output
            n_k1 = len(np.unique(y[:, output_idx[0]]))
            n_k2 = len(np.unique(y[:, output_idx[1]]))

            if n_k1 < n_k2:
                min_k, max_k = output_idx[0], output_idx[1]
            else:
                min_k, max_k = output_idx[1], output_idx[0]

            output_idx = min_k, max_k


            # Get the count results for train & test data
            train_data = make_counts(y, output_idx, train_idx)
            test_data = make_counts(y, output_idx, test_idx)
            
            diff_data = abs(train_data - test_data)
            diff_values_medians.append( np.median(diff_data.values) )
            diff_values_stds.append( diff_data.values.std() )



        # Store the best average result
        diff_result = np.mean(diff_values_medians + diff_values_stds)
        print('RESULT: ', np.round(diff_result, 5), '\n\n')

        if diff_result < best_result:
            best_result = diff_result
            best_params = p.copy()
            best_splitter = stratified_splitter
        



    print(15 * '...')
    print('BEST RESULT: ', np.round(best_result, 5))
    print('BEST PARAMS: ', best_params)
    return best_splitter, best_params




