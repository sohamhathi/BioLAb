import random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Load dataset ---
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# --- Objective function: we want to maximize accuracy (minimize 1 - accuracy) ---
def fitness_function(params):
    C, gamma = params
    model = SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy  # PSO minimizes this


# --- Particle class ---
class Particle:
    def __init__(self, bounds):
        # C, gamma ranges
        self.position = [
            random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1])
        ]
        self.velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.best_pos = list(self.position)
        self.best_val = fitness_function(self.position)

    def update_velocity(self, global_best_pos, w=0.5, c1=1.5, c2=1.5):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_pos[i] - self.position[i])
            social = c2 * r2 * (global_best_pos[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            # Keep position in bounds
            self.position[i] = max(bounds[i][0], min(bounds[i][1], self.position[i]))


# --- PSO algorithm ---
def particle_swarm_optimization(num_particles=20, max_iter=30):
    bounds = [(0.1, 100), (0.0001, 1)]  # (C range), (gamma range)
    swarm = [Particle(bounds) for _ in range(num_particles)]

    global_best = min(swarm, key=lambda p: p.best_val)
    global_best_pos = list(global_best.best_pos)
    global_best_val = global_best.best_val

    for iteration in range(max_iter):
        for particle in swarm:
            fitness = fitness_function(particle.position)
            if fitness < particle.best_val:
                particle.best_val = fitness
                particle.best_pos = list(particle.position)

            if fitness < global_best_val:
                global_best_val = fitness
                global_best_pos = list(particle.position)

        for particle in swarm:
            particle.update_velocity(global_best_pos)
            particle.update_position(bounds)

        print(f"Iteration {iteration+1}/{max_iter} | Best Accuracy = {(1 - global_best_val):.4f}")

    return global_best_pos, global_best_val


# --- Run the program ---
if __name__ == "__main__":
    best_pos, best_val = particle_swarm_optimization()
    print("\n✅ Optimization complete!")
    print(f"Best Parameters: C = {best_pos[0]:.4f}, gamma = {best_pos[1]:.4f}")
    print(f"Best Accuracy: {(1 - best_val):.4f}")

