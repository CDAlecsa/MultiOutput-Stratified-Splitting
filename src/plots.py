# Load modules
import os

import numpy as np
import pandas as pd
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.animation import (FuncAnimation, 
                                  PillowWriter)

import seaborn as sns
sns.set_theme()

from typing import Union
from itertools import combinations

from misc import INTP, make_counts
from sklearn.utils import shuffle







def plot_losses(losses: list, 
                stds: list,
                save_path: Union[str, None] = None):
    '''
        Plot the values of the fitness function.
    '''
    plt.plot(range(len(losses)), losses, marker = 'o', color = 'darkorange')
    plt.fill_between(range(len(losses)), 
                        np.maximum(losses - stds, [0]), 
                        losses + stds, 
                        facecolor = 'blue', 
                        alpha = 0.25)
    plt.title('Fitness function')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "fitness.jpg"), dpi = 300)
        plt.close()
    else:
        plt.show()





def generate_axes(y: npt.NDArray[INTP], 
                  random_state: Union[None, int, np.random.RandomState]):
    '''
        Function which generates `matplotlib` axes with respect to the number of outputs.
        When `n_outputs` is high dimensional, then we choose only 4 target outputs for the visual representation.
    '''
    
    n_outputs = y.shape[1]
    indices = list(combinations(range(n_outputs), r = 2))

    if n_outputs == 2:
        n_rows, n_cols = 1, 1
    elif n_outputs == 3:
        n_rows, n_cols = 2, 2
    elif n_outputs >= 4:
        n_rows, n_cols = 3, 2
        indices = shuffle(indices, random_state = random_state)

    # Generate axes
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (16, 10))
    axs = [axs] if n_rows == 1 and n_cols == 1 else axs.ravel()
    axs = axs[:len(indices)]

    return fig, axs, indices





def plot_counts(y: npt.NDArray[INTP], 
                data_indices: list, 
                random_state: Union[None, int, np.random.RandomState], 
                title: str,
                save_path: Union[str, None] = None):
    '''
        Plot the counts corresponding to each target output.
    '''

    assert ( len(data_indices) == 2 )
    train_idx, test_idx = data_indices

    diff_values_medians, diff_values_stds = [], []
        
    
    # Generate axes
    _, axs, indices = generate_axes(y, random_state)


    # Make plots
    for i, ax in enumerate(axs):
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
        diff_values_medians.append( np.round(np.median(diff_data.values), 3) )
        diff_values_stds.append( np.round(diff_data.values.std(), 3) )


        # Make annotations for the final heatmap
        annotations = np.char.add(np.full(shape = test_data.shape, fill_value = "%] / ["), test_data.values.astype(str))
        annotations = np.char.add(annotations, np.full(shape = test_data.shape, fill_value = "]%"))

        annotations = (np.asarray(["[{1:.1f} {0}".format(string, value)
                    for string, value in zip(annotations.flatten(),
                                            train_data.values.flatten())])
        ).reshape(test_data.shape)
    

        # Plot heatmap for the current output pair
        sns.heatmap(diff_data.astype(float), annot = annotations, linewidth = 0.75, cmap = "YlGnBu", fmt = "", ax = ax)
        ax.set_xlabel('y_' + str(output_idx[0]), fontsize = 15)
        ax.set_ylabel('y_' + str(output_idx[1]), fontsize = 15)


    plt.suptitle( title + ': ' + str(diff_values_medians) + ' - ' + str(diff_values_stds))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "counts_percentages.jpg"), dpi = 300)
        plt.close()
    else:
        plt.show()





def plot_distributions(y: npt.NDArray[INTP], 
                       data_indices: list,
                       losses: list, 
                       random_state: Union[None, int, np.random.RandomState], 
                       title: str,
                       save_path: Union[str, None] = None):
    '''
        Plot the distribution of the target counts with respect to each output.
    '''

    assert ( len(data_indices) == 2 )
    train_idx, test_idx = data_indices
    
    # Generate axes
    _, axs, indices = generate_axes(y, random_state)

    # Retrieve the partitioning made by the genetic algorithm
    y_train = y[train_idx]
    y_test = y[test_idx]


    # Make plots
    for i, ax in enumerate(axs):
        k1, k2 = indices[i]

        y_k1 = y[:, k1]
        y_k2 = y[:, k2]

        y_train_k1 = y_train[:, k1]
        y_train_k2 = y_train[:, k2]

        y_test_k1 = y_test[:, k1]
        y_test_k2 = y_test[:, k2]

        df_train = pd.DataFrame( data = {'y_' + str(k1): y_train_k1, 'y_'  + str(k2): y_train_k2}, dtype = int )
        df_test = pd.DataFrame( data = {'y_' + str(k1): y_test_k1, 'y_'  + str(k2): y_test_k2}, dtype = int )

        n_k1 = len(np.unique(y_k1))
        n_k2 = len(np.unique(y_k2))

        if n_k1 < n_k2:
            min_k, max_k = k1, k2
        else:
            min_k, max_k = k2, k1

        sns.countplot(df_train, x = "y_" + str(max_k), hue = "y_" + str(min_k), ax = ax, edgecolor = 'green', linewidth = 2.5, 
                      stat = 'percent')
        sns.countplot(df_test, x = "y_" + str(max_k), hue = "y_" + str(min_k), ax = ax, edgecolor = 'blue', linewidth = 2.5, 
                      legend = False, 
                      stat = 'percent')        
    

    plt.suptitle(title + ' - [fitness: ' + str( round(losses[-1], 3)) + ']' )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "distribution_counts.jpg"), dpi = 300)
        plt.close()
    else:
        plt.show()





def animate(iteration: int, 
            axs: object,
            y: npt.NDArray[INTP], 
            indices: list[tuple],
            train_indices: list, 
            test_indices: list,
            losses: list,
            title: str):
    '''
        Function which makes the animation of the counts distributions at a given iteration.
    '''

    # Retrieve the partitioning made by the genetic algorithm
    y_train = y[ train_indices[iteration] ]
    y_test = y[ test_indices[iteration] ]


    # Make plots
    for i, ax in enumerate(axs):
        ax.clear()

        k1, k2 = indices[i]

        y_k1 = y[:, k1]
        y_k2 = y[:, k2]

        y_train_k1 = y_train[:, k1]
        y_train_k2 = y_train[:, k2]

        y_test_k1 = y_test[:, k1]
        y_test_k2 = y_test[:, k2]

        df_train = pd.DataFrame( data = {'y_' + str(k1): y_train_k1, 'y_'  + str(k2): y_train_k2}, dtype = int )
        df_test = pd.DataFrame( data = {'y_' + str(k1): y_test_k1, 'y_'  + str(k2): y_test_k2}, dtype = int )

        n_k1 = len(np.unique(y_k1))
        n_k2 = len(np.unique(y_k2))

        if n_k1 < n_k2:
            min_k, max_k = k1, k2
        else:
            min_k, max_k = k2, k1

        sns.countplot(df_train, x = "y_" + str(max_k), hue = "y_" + str(min_k), ax = ax, edgecolor = 'green', linewidth = 2.5, 
                      stat = 'percent')
        sns.countplot(df_test, x = "y_" + str(max_k), hue = "y_" + str(min_k), ax = ax, edgecolor = 'blue', linewidth = 2.5, legend = False, 
                      stat = 'percent')        
    
    
    plt.suptitle(title + '[iteration: ' + str(iteration) + '] & [fitness: ' + str( round(losses[iteration], 3)) + ']' )
    plt.tight_layout()
    





def make_gif_distributions(y: npt.NDArray[INTP], 
                           data_indices: list,
                           losses: list, 
                           random_state: Union[None, int, np.random.RandomState], 
                           title: str,
                           gif_fps: int,
                           animation_interval: int,
                           save_path: str):
    '''
        Make a gif for the animation of the target counts with respect to genetic algorithm's iterations.
    '''

    assert ( len(data_indices) == 2 )
    train_indices, test_indices = data_indices

    # Generate axes
    fig, axs, indices = generate_axes(y, random_state)

    # Make matplotlib animation
    animation_fct = lambda i: animate(i, axs, y, indices, train_indices, test_indices, losses, title)
    anim = FuncAnimation(fig, animation_fct, frames = len(losses), interval = animation_interval, repeat = True)
    
    writer_gif = PillowWriter(fps = gif_fps) 
    anim.save(os.path.join(save_path, "distribution_counts_animation.gif"), writer = writer_gif)


