import time

import numpy as np


def LSO(positions, objective_function, lb, ub,  max_iter):

    #Light Spectrum Optimizer (LSO) algorithm implementation in Python

    # Initialization
    dimension, num_particles = positions.shape[0], positions.shape[1]
    # positions = np.random.uniform(low=lb, high=ub, size=(num_particles, dimension))
    velocities = np.zeros_like(positions)
    intensities = np.zeros(num_particles)
    best_intensities = np.zeros(num_particles)
    best_positions = np.zeros_like(positions)

    # Evaluate initial positions
    for i in range(num_particles):
        intensities[i] = objective_function(positions[i])
        best_intensities[i] = intensities[i]
        best_positions[i] = positions[i].copy()

    # Find global best position and intensity
    gbest_idx = np.argmin(intensities)
    gbest_score = intensities[gbest_idx]
    gbest_pos = positions[gbest_idx].copy()

    Convergence_curve = np.zeros(max_iter)
    ct = time.time()
    # LSO main loop
    for t in range(max_iter):
        # Update velocities and positions
        for i in range(num_particles):
            # Update velocity
            for j in range(dimension):
                c1 = np.random.uniform()
                c2 = np.random.uniform()
                velocities[i, j] = velocities[i, j] + c1 * (best_positions[i, j] - positions[i, j]) + c2 * (
                            gbest_pos[j] - positions[i, j])

            # Update position
            positions[i] = positions[i] + velocities[i]

            # Boundary checking
            positions[i] = np.maximum(positions[i], lb)
            positions[i] = np.minimum(positions[i], ub)

        # Evaluate new positions
        for i in range(num_particles):
            intensities[i] = objective_function(positions[i])

            # Update personal best
            if intensities[i] < best_intensities[i]:
                best_intensities[i] = intensities[i]
                best_positions[i] = positions[i].copy()

                # Update global best
                if intensities[i] < gbest_score:
                    gbest_score = intensities[i]
                    gbest_pos = positions[i].copy()

        Convergence_curve[t] = intensities
    ct = time.time() - ct
    return gbest_score, gbest_pos, Convergence_curve, ct


