import random
import time

import numpy as np


def HHOA(Positions, fitness_func, lb, ub, max_iter):

    # Initialization
    dim, num_horses = Positions.shape[0], Positions.shape[1]
    fitness = np.zeros(num_horses)
    best_fitness = float("inf")
    best_solution = np.zeros(dim)

    for i in range(num_horses):
        Positions[i] = np.random.uniform(lb, ub, dim)
        fitness[i] = fitness_func(Positions[i])
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_solution = np.copy(Positions[i])

    Convergence_curve = np.zeros(max_iter)
    ct = time.time()
    # Main loop
    for t in range(max_iter):
        a = 2 - 2 * t / max_iter  # shrinking encircling mechanism coefficient
        r = np.random.uniform(0, 1)  # randomization parameter

        for i in range(num_horses):
            for j in range(dim):
                # Horse encircling mechanism
                c = np.sum(Positions, axis=0) / num_horses
                if r < 0.5:
                    Positions[i, j] = c[j] - a * abs(c[j] - Positions[i, j])
                else:
                    Positions[i, j] = c[j] + a * abs(c[j] - Positions[i, j])

                # Bounds handling
                if Positions[i, j] < lb:
                    Positions[i, j] = lb
                elif Positions[i, j] > ub:
                    Positions[i, j] = ub

            # Fitness evaluation
            fitness[i] = fitness_func(Positions[i])
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_solution = np.copy(Positions[i])

            # Horse attacking mechanism
            for k in range(num_horses):
                if fitness[i] < fitness[k]:
                    Positions[k] += np.random.uniform(0, 1) * (Positions[i] - Positions[k])
        Convergence_curve[t] = fitness
    ct = time.time() - ct
    return best_fitness, best_solution, Convergence_curve, ct



