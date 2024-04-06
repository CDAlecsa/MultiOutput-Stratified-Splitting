# Load modules
import os
from datetime import datetime

from dataloader import CustomDataset
from splitter import (MultiOutputStratifiedSplitter, 
                      grid_search_param)

from misc import save_dict
from plots import (plot_losses, 
                   plot_counts,
                   plot_distributions, 
                   make_gif_distributions)






# The folder in which we save the results
save_path = os.path.join("..", "results")
os.makedirs(save_path, exist_ok = True)

save_path = os.path.join(save_path, datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
os.makedirs(save_path, exist_ok = True)






# Data-related parameters
n_samples = 10000
n_outputs = 4
test_size = 0.33
random_state = 0

# Genetic algorithm related parameters
verbose = True
population_size = 200
n_individuals_by_crossover = 40
n_individuals_by_mutation = 40
sample_with_replacement = True

# Parameter used for making the gif corresponding to the evolution 
# of the train/test counts obtained from genetic algorithm's iterations
make_gif = True

# Parameter used for showing the plot corresponding to the synthetic data
show_data = False

# Parameter which runs the grid search param
make_grid_search = True

# Method representation
method_name = 'MultiOutputStratifiedSplitter'





# Generate data
data = CustomDataset(n_samples = n_samples, 
                     n_outputs = n_outputs,
                     random_state = random_state)
X, y = data.generate_data()

# Plot the data
if show_data:
    data.plot(y)





# Train the stratifier
if make_grid_search:
    n_iterations = 20
    mutation_rate = [0.15, 0.85, 5]
    crossover_rate = [1, 50, 10]
    stratified_splitter, best_params = grid_search_param(X = X,
                                                         y = y,
                                                         random_state = random_state,
                                                         test_size = test_size,
                                                         n_iterations = n_iterations,
                                                         population_size = population_size,
                                                         mutation_rate = mutation_rate,
                                                         crossover_rate = crossover_rate,
                                                         n_individuals_by_mutation = n_individuals_by_mutation,
                                                         n_individuals_by_crossover = n_individuals_by_crossover,
                                                         sample_with_replacement = sample_with_replacement
                                                    )
    best_params['random_state'] = random_state
    best_params['n_samples'] = n_samples
    best_params['n_outputs'] = n_outputs
    best_params['make_grid_search'] = make_grid_search

else:

    n_iterations = 150
    mutation_rate = 0.65
    crossover_rate = 0.35                               
    stratified_splitter = MultiOutputStratifiedSplitter(n_iterations = n_iterations,
                                                        test_size = test_size,
                                                        population_size = population_size,
                                                        mutation_rate = mutation_rate, 
                                                        crossover_rate = crossover_rate,
                                                        n_individuals_by_mutation = n_individuals_by_mutation,
                                                        n_individuals_by_crossover = n_individuals_by_crossover,
                                                        sample_with_replacement = sample_with_replacement,
                                                        verbose = verbose,
                                                        random_state = random_state
                                                        )

    X_train, X_test, y_train, y_test = stratified_splitter.fit(X, y)

    best_params = dict()
    best_params['n_iterations'] = n_iterations
    best_params['test_size'] = test_size
    best_params['population_size'] = population_size
    best_params['mutation_rate'] = mutation_rate
    best_params['crossover_rate'] = crossover_rate
    best_params['n_individuals_by_mutation'] = n_individuals_by_mutation
    best_params['n_individuals_by_crossover'] = n_individuals_by_crossover
    best_params['sample_with_replacement'] = sample_with_replacement
    best_params['random_state'] = random_state
    best_params['n_samples'] = n_samples
    best_params['n_outputs'] = n_outputs
    best_params['make_grid_search'] = make_grid_search




# Plot the stratifier's characteristics
plot_losses(losses = stratified_splitter.losses, 
            stds = stratified_splitter.stds,
            save_path = save_path)

# Plot the counts matrix associated with the stratified splitting
plot_counts(y = y, 
            data_indices = [stratified_splitter.train_idx, stratified_splitter.test_idx], 
            random_state = stratified_splitter.random_state,
            title = method_name,
            save_path = save_path
            )


# Plot the distribution counts associated with the stratified splitting
plot_distributions(y = y, 
                   data_indices = [stratified_splitter.train_idx, stratified_splitter.test_idx], 
                   losses = stratified_splitter.losses,
                   random_state = stratified_splitter.random_state,
                   title = method_name,
                   save_path = save_path
                  )


# Make the animation related to the distribution counts associated with the stratified splitting
if make_gif:
    make_gif_distributions(y = y, 
                           data_indices = [stratified_splitter.train_indices, stratified_splitter.test_indices], 
                           losses = stratified_splitter.losses,
                           random_state = stratified_splitter.random_state,
                           title = method_name,
                           gif_fps = n_iterations // 20,
                           animation_interval = 200,
                           save_path = save_path
                        )


# Save the parameters corresponding to the current simulation
save_dict(best_params, 
          save_path = save_path)

