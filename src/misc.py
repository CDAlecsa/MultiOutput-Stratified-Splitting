# Load modules
import os, time, json

import numpy as np
import pandas as pd
import numpy.typing as npt

from functools import wraps


# Numpy types
from numpy import intp as INTP
from numpy import float32 as FLOAT32







def timeit(func: callable):
    '''
        Python decorator for measuring the computational time.
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function [{func.__name__}]: [{total_time:.4f} seconds].')
        return result
    return timeit_wrapper





def make_counts(y: npt.NDArray[INTP], 
                output_idx: tuple, 
                data_idx: npt.NDArray[INTP]):
    '''
        Make the DataFrame corresponding to the target train/test subsets.
    '''

    # Define the DataFrame corresponding to the pair counts
    df = pd.DataFrame( data = { 'y_' + str(k): y[data_idx, k] for k in output_idx } )
    counts = df[ [ 'y_' + str(k) for k in output_idx ] ].value_counts(normalize = True).reset_index(name = 'count')

    # Compute the percentages of counts
    counts['count'] *= 100

    # Compute the unique value with respect to each output
    subset_unique_values = { 'y_' + str(k): counts['y_' + str(k)].unique() for k in output_idx }

    # Store the string representation of the indices
    str_idx_min = str(output_idx[0])
    str_idx_max = str(output_idx[1])

    # Create the new DataFrame which will be used in the heatmap
    data = pd.DataFrame(columns = { str(c): c for c in subset_unique_values['y_' + str_idx_min] }, 
                        index = subset_unique_values['y_' + str_idx_max] )
    
    # Compute the values of the aforementioned heatmap using the pair counts
    for i in subset_unique_values['y_' + str_idx_max]:
        for j in subset_unique_values['y_' + str_idx_min]:
            current_counts = counts[ (counts['y_' + str_idx_min] == j) & (counts['y_' + str_idx_max] == i) ]['count'].values
            data.iloc[i, j] = current_counts.item() if len(current_counts) > 0 else 0.0
            data.iloc[i, j] = np.round(data.iloc[i, j], 2)


    # Sort data with respect to each output value
    data = data.sort_index(axis = 0, inplace = False)
    data = data.sort_index(axis = 1, inplace = False)
    return data





# Save the hyper-parameters dictionary into a json file
def save_dict(d: dict, save_path: str):
    with open(os.path.join(save_path, "parameters.json"), "w") as outfile: 
        json.dump({k : str(v) for (k, v) in d.items()}, 
                  outfile, 
                  indent = 4
                  )





