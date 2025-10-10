import numpy as np

# Step 1: Define cities (x, y coordinates)
cities = np.array([
    [1, 1],
    [4, 1],
    [4, 5],
    [1, 5]
])

num_cities = len(cities)

# Step 2: Initialize parameters
num_ants = 10
alpha = 1          # pheromone importance
beta = 5           # heuristic importance
rho = 0.5          # evaporation rate
pheromone_init = 0.1
iterations = 20    # Reduced for readability
Q = 1              # pheromone deposit factor

# Compute distance matrix
def distance_matrix(cities):
    dist = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist

dist_matrix = distance_matrix(cities)

# Heuristic matrix (inverse distance), avoid division by zero
heuristic = 1 / (dist_matrix + 1e-10)

# Initialize pheromone matrix
pheromone = np.ones((num_cities, num_cities)) * pheromone_init

def select_next_city(current_city, visited):
    probabilities = []
    for city in range(num_cities):
        if city not in visited:
            tau = pheromone[current_city][city] ** alpha
            eta = heuristic[current_city][city] ** beta
            probabilities.append(tau * eta)
        else:
            probabilities.append(0)
    probabilities = np.array(probabilities)
    probabilities_sum = probabilities.sum()
    if probabilities_sum == 0:
        # If no options, choose randomly from unvisited
        unvisited = [c for c in range(num_cities) if c not in visited]
        return np.random.choice(unvisited)
    probabilities /= probabilities_sum
    return np.random.choice(range(num_cities), p=probabilities)

def construct_solution():
    solutions = []
    lengths = []
    for _ in range(num_ants):
        visited = []
        start_city = np.random.randint(num_cities)
        visited.append(start_city)

        while len(visited) < num_cities:
            current_city = visited[-1]
            next_city = select_next_city(current_city, visited)
            visited.append(next_city)

        visited.append(visited[0])  # return to start
        solutions.append(visited)

        # Calculate route length
        length = 0
        for i in range(len(visited) - 1):
            length += dist_matrix[visited[i]][visited[i + 1]]
        lengths.append(length)

    return solutions, lengths

def update_pheromone(solutions, lengths):
    global pheromone
    # Evaporate pheromone
    pheromone = (1 - rho) * pheromone

    # Deposit pheromone
    for i, route in enumerate(solutions):
        length = lengths[i]
        for j in range(len(route) - 1):
            city_i = route[j]
            city_j = route[j + 1]
            pheromone[city_i][city_j] += Q / length
            pheromone[city_j][city_i] += Q / length  # symmetric TSP

best_route = None
best_length = float('inf')

for iteration in range(iterations):
    solutions, lengths = construct_solution()
    update_pheromone(solutions, lengths)

    min_length = min(lengths)
    min_index = lengths.index(min_length)

    if min_length < best_length:
        best_length = min_length
        best_route = solutions[min_index]

    # Detailed iteration output
    print(f"Iteration {iteration + 1}:")
    print(f"  Best route this iteration: {solutions[min_index]}")
    print(f"  Length of best route this iteration: {min_length:.4f}")
    print(f"  Global best route so far: {best_route}")
    print(f"  Length of global best route so far: {best_length:.4f}\n")

print("Final Best Route:", best_route)
print("Final Best Route Length:", best_length)
