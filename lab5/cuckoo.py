import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Objective function: return error rate for given SVM hyperparameters
def svm_error_rate(params):
    C, gamma = params
    # Make sure parameters are within bounds
    if C <= 0 or gamma <= 0:
        return 1.0  # Max error
    
    svm = SVC(C=C, gamma=gamma)
    # Use 5-fold cross-validation accuracy
    scores = cross_val_score(svm, X, y, cv=5)
    error = 1 - scores.mean()  # Minimize error
    return error

# Lévy flight function (fixed)
def levy_flight(Lambda, size):
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def cuckoo_search(objective_func, n=20, dim=2, lb=[0.1, 0.0001], ub=[100, 1], pa=0.25, max_iter=100):
    lb = np.array(lb)
    ub = np.array(ub)
    
    nests = lb + (ub - lb) * np.random.rand(n, dim)
    fitness = np.array([objective_func(nest) for nest in nests])
    
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    Lambda = 1.5

    for iteration in range(max_iter):
        for i in range(n):
            step = levy_flight(Lambda, dim)
            step_size = 0.01 * step * (nests[i] - best_nest)
            new_nest = nests[i] + step_size
            new_nest = np.clip(new_nest, lb, ub)
            new_fitness = objective_func(new_nest)
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_nest = new_nest.copy()

        # Abandon worst nests and create new ones
        K = np.random.rand(n, dim) > pa
        new_nests = lb + (ub - lb) * np.random.rand(n, dim)
        nests = nests * K + new_nests * (1 - K)
        fitness = np.array([objective_func(nest) for nest in nests])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()

        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration}: Best error rate = {best_fitness:.4f}")

    return best_nest, best_fitness

# Run hyperparameter tuning
best_params, best_error = cuckoo_search(svm_error_rate)
print("\nBest hyperparameters found:")
print(f"C = {best_params[0]:.4f}, gamma = {best_params[1]:.6f}")
print(f"Cross-validated error rate = {best_error:.4f}")
