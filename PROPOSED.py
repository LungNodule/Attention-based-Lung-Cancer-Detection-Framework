import time
import numpy as np

# Election-Based Optimization Algorithm
def PROPOSED(population_size, fobj, lb, ub, max_iterations):
    # Parameters
    SearchAgents_no, dim = population_size.shape[0], population_size.shape[1]
    population_size = SearchAgents_no
    num_dimensions = dim
    mutation_rate = 0.1

    # Initialize population
    population = np.random.rand(population_size, num_dimensions) * (ub - lb) + lb
    fitness = np.zeros(population_size)

    # Initialize best solution and convergence curve
    best_score = float('inf')
    best_voltage = np.zeros(num_dimensions)
    Convergence_curve = np.zeros(max_iterations)
    ct = time.time()
    # Main loop
    for iteration in range(max_iterations):
        # Evaluate fitness of the population
        for i in range(population_size):
            fitness[i] = fobj(population[i, :])

        # Election process: Select the leader (best solution)
        leader_index = np.argmin(fitness)
        leader = population[leader_index, :]
        leader_fitness = fitness[leader_index]

        # Campaign: Update the population based on the leader
        for i in range(population_size):
            if i != leader_index:  # Exclude the leader
                # Perform crossover and mutation with the leader
                Lambda =  min(fitness) / np.sqrt(min(fitness) ^ 2 + max(fitness) ^ 2)

                new_solution = np.concatenate((population[i, :Lambda], leader[Lambda:]))

                # Mutation
                for j in range(num_dimensions):
                    if np.random.rand() < mutation_rate:
                        new_solution[j] = new_solution[j] + np.random.randn()

                # Ensure the new solution adheres to lb and ub
                new_solution = np.maximum(np.minimum(new_solution, ub), lb)

                # Evaluate the fitness of the new solution
                new_fitness = fobj(new_solution)

                # Replace the old solution if the new solution is better
                if new_fitness < fitness[i]:
                    population[i, :] = new_solution
                    fitness[i] = new_fitness

        # Update convergence curve
        Convergence_curve[iteration] = leader_fitness

    # Find the best solution
    best_index = np.argmin(fitness)
    best_score = fitness[best_index]
    best_voltage = population[best_index, :]
    ct = time.time() - ct
    return best_voltage, Convergence_curve, best_score, ct
