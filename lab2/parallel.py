import numpy as np
import random

def fitness(x):
    return x * np.sin(x)

grid_size = (5, 5)   
iterations = 20
lower_bound, upper_bound = 0, 10
mutation_rate = 0.1

grid = np.random.uniform(lower_bound, upper_bound, grid_size)

def get_neighbors(grid, i, j):
    neighbors = []
    rows, cols = grid.shape
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = (i + di) % rows, (j + dj) % cols  
            neighbors.append(grid[ni, nj])
    return neighbors

best_solution = None
best_fitness = float("-inf")

for it in range(iterations):
    new_grid = grid.copy()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            current = grid[i, j]
            neighbors = get_neighbors(grid, i, j)
            candidates = neighbors + [current]

          
            best_neighbor = max(candidates, key=fitness)

           
            if random.random() < mutation_rate:
                mutated = best_neighbor + np.random.uniform(-0.5, 0.5)
                mutated = np.clip(mutated, lower_bound, upper_bound)
                best_neighbor = mutated

            new_grid[i, j] = best_neighbor

            
            if fitness(best_neighbor) > best_fitness:
                best_fitness = fitness(best_neighbor)
                best_solution = best_neighbor

    grid = new_grid
    print(f"Iteration {it+1}: Best solution so far = {best_solution:.4f}, Fitness = {best_fitness:.4f}")


print("\nFinal Best Solution:", best_solution)
print("Final Best Fitness:", best_fitness)
