# Load modules
import math

import numpy as np
import numpy.typing as npt

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from misc import INTP, FLOAT32
from plots import generate_axes

from typing import Union
from warnings import warn
from numbers import Integral
from itertools import product

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import (Interval, 
                                             validate_parameter_constraints)







########### The class representing the generation of a synthetic dataset ###########
class Dataset():


    # Constraints
    _parameter_constraints: dict = {
        "n_samples": [Interval(Integral, 2, None, closed = "left")],
        "n_features": [Interval(Integral, 1, None, closed = "left")], 
        "n_classes": [list],
        "random_state": ["random_state"]
    }




    def __init__(self, 
                 n_samples: int, 
                 n_features: int, 
                 n_classes: list[int], 
                 random_state: Union[None, int, np.random.RandomState] = None):
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state

        self.check_params()
        self.n_outputs = len(self.n_classes)
        
        


        
    def get_params(self):
        '''
            Retrieves a dictionary corresponding to the class parameters.
        '''
        _params = { k: v for (k, v) in zip( self._parameter_constraints.keys(), [self.n_samples, 
                                                                                 self.n_features, 
                                                                                 self.n_classes,
                                                                                 self.random_state] ) }
        return _params




    def check_params(self):
        '''
            Check the initialization values for errors
        '''
        
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(),
            caller_name = self.__class__.__name__
        )

        if len(self.n_classes) < 2:
            raise ValueError("Since we are dealing with multioutput tasks, we must have at least 2 outputs.")
        if not all(isinstance(x, Integral) for x in self.n_classes):
            raise TypeError("Each element of the `n_classes` argument must be of type `int`.")





    @staticmethod
    def correct_proba(proba: list[float]):
        ''' 
            Make the probabilities sum to 1.
        '''
    
        if sum(proba) != 1.0:
            proba[-1] = max(0, 1 - np.sum(proba[0:-1])) 
        return proba





    def generate_target_proba(self, n_cls: int, p_dict: dict[int, float]):
        """
            Function which generates a probability sample which will be used for creating a 1D target vector.
        """

        # Default case: no new probabilities are given
        if p_dict is None:
            proba = {k: 1/n_cls for k in range(n_cls)}


        # Otherwise: determine all the probabilities from the ones that are given
        else:

            # Retrieve the number of given probabilities
            len_p = len(p_dict)

            # Check if all the given probabilities are in [0.0, 1.0]:
            if not all([0.0 <= p <= 1.0 for p in p_dict.values()]):
                raise ValueError("Some probabilities are not well defined.")

            # Check if the indices of the given dictionary are correct
            if not all([i in range(n_cls) for i in p_dict.keys() ]):
                raise KeyError("The keys of the given dictionary must belong to the indices of `n_classes`.")

            

            # Analyzing cases corresponding to the given probabilities and the number of classes
            if len_p > n_cls:
                warn("Ignoring the last {} probabilities. "
                     "There are given {} probabilities and {} n_classes.".format(len_p - n_cls, len_p, n_cls)
                     )
                p_dict = dict( list(p_dict.items())[ : n_cls] )
            

            sum_p = sum(p_dict.values())

            if len_p == n_cls and sum_p != 1.0:
                raise ValueError("The given probabilities must sum to 1.0.")
            if len_p < n_cls and not (0.0 < sum_p < 1.0):
                raise ValueError("The sum of probabilies is not in (0.0, 1.0).")



            # Generating the full probability sample
            proba = {k: None for k in range(n_cls)}
            remaining_idx = [ i for i in range(n_cls) if i not in p_dict.keys() ]
            
            for (k, v) in p_dict.items():
                proba[k] = v
                
            for k in remaining_idx:
                proba[k] = (1 - sum_p) / len(remaining_idx)

        return proba





    def generate_joint_proba(self, p_list: list[dict[int, float]]):
        """
            Function which generates a probability sample for multiple output dimensions.
        """
        proba_classes = self.n_outputs * [None]

        # Generate the probabilities for each target
        for i in range(self.n_outputs):
            proba_classes[i] = self.generate_target_proba(self.n_classes[i], p_list[i])

        # Range variables
        idx_ranges = [ range(self.n_classes[i]) for i in range(self.n_outputs) ]
        proba_ranges = [ list(proba_classes[i].values()) for i in range(self.n_outputs) ]

        # Generate joint probabilities
        indices = list(product( *idx_ranges ))
        proba = list(product( *proba_ranges ))
        proba = [math.prod(p) for p in proba]

        # Make the probabilities sum to 1
        proba = self.correct_proba(proba)
        return indices, proba





    def generate_multioutput_target(self, p_list: list[dict[int, float]]):
        '''
            The actual function in which we generate the multioutput target.
        '''
        indices, proba = self.generate_joint_proba(p_list)
        X, y = self.initialize_data()

        rng = check_random_state(self.random_state)
        
        z = rng.choice( range(len(indices)), self.n_samples, p = proba, replace = True )
        z = [indices[c] for c in z]

        for k in range(self.n_outputs):
            y[:, k] = [i[k] for i in z]

        return X, y

    



    def initialize_data(self):
        '''
            Initialize the values for the data pair (X, y).
        '''
        shape_X = (self.n_samples, self.n_features)
        shape_y = (self.n_samples, self.n_outputs)
        
        X: npt.NDArray[FLOAT32] = np.empty(shape = shape_X, dtype = FLOAT32)
        y: npt.NDArray[INTP] = np.empty(shape = shape_y, dtype = INTP)
        return X, y





    def generate_data(self, p_list: list[dict[int, float]]):
        '''
            Generate the dataset pair (X, y), where y is a multi-output target.
        '''
        len_p_list = len(p_list)
        n_outputs_ = self.n_outputs

        if n_outputs_ != len_p_list:
            raise ValueError("Number of outputs {} "
                             "must match the length {} of the list "
                             "containing the probabilities".format(n_outputs_, len_p_list))

        X, y = self.generate_multioutput_target(p_list)
        return X, y





    def plot(self, 
             y: npt.NDArray[INTP]):
        '''
            Make plots regarding the distribution of target counts.
        '''

        # Generate axes
        axs, indices = generate_axes(y, self.random_state)
        
        # Make plots
        for i, ax in enumerate(axs):
            k1, k2 = indices[i]

            y_k1 = y[:, k1]
            y_k2 = y[:, k2]

            df = pd.DataFrame( data = {'y_' + str(k1): y_k1, 'y_'  + str(k2): y_k2}, dtype = int )

            n_k1 = len(np.unique(y_k1))
            n_k2 = len(np.unique(y_k2))

            if n_k1 < n_k2:
                min_k, max_k = k1, k2
            else:
                min_k, max_k = k2, k1

            sns.countplot(df, x = "y_" + str(max_k), hue = "y_" + str(min_k), ax = ax)        

        plt.tight_layout()
        plt.show()







########### Examples of custom datasets ###########
class CustomDataset(Dataset):

    def __init__(self, 
                 n_samples: int = 1000,
                 n_features: int = 100,
                 n_outputs: int = 2,
                 random_state: Union[None, int, np.random.RandomState] = None):

        if n_outputs == 2:
            n_classes = [5, 7]
        elif n_outputs == 3:
            n_classes = [5, 2, 3]
        elif n_outputs == 4:
            n_classes = [2, 2, 3, 3]
        
        super().__init__(n_samples, n_features, n_classes, random_state)
        assert self.n_outputs == n_outputs





    def generate_data(self):

        if self.n_outputs == 2:
            p_list = [  {0: 0.25, 2: 0.15, 4: 0.35},
                        {0: 0.035, 1: 0.1, 2: 0.055, 3: 0.15, 4: 0.1, 5: 0.2}
            ]
        elif self.n_outputs == 3:
            p_list = [  {0: 0.25, 1: 0.1, 2: 0.16, 3: 0.4},
                        {1: 0.85},
                        {1: 0.25, 2: 0.35}
            ]
        elif self.n_outputs == 4:
            p_list = [  {0: 0.35},
                        {1: 0.55},
                        {1: 0.2, 2: 0.3},
                        {0: 0.15, 1: 0.45}
            ]

        return super().generate_data(p_list)
        
