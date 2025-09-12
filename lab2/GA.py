import numpy as np
import random

def fitness(x):
    return x * np.sin(4 * x) + np.cos(2 * x)

POP_SIZE = 20
GENE_LENGTH = 8      
LOWER_BOUND, UPPER_BOUND = 0, 10
GENERATIONS = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7


def create_population():
    return ["".join(random.choice("01") for _ in range(GENE_LENGTH)) for _ in range(POP_SIZE)]


def decode_gene(gene):
    value = int(gene, 2) 
    scaled = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * value / (2**GENE_LENGTH - 1)
    return scaled

def selection(population):
    k = 3
    selected = random.sample(population, k)
    return max(selected, key=lambda g: fitness(decode_gene(g)))


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENE_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2


def mutation(gene):
    gene_list = list(gene)
    for i in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            gene_list[i] = "1" if gene_list[i] == "0" else "0"
    return "".join(gene_list)

def gene_expression_algorithm():
    population = create_population()
    best_gene, best_fit = None, float("-inf")

    for gen in range(GENERATIONS):
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutation(c1), mutation(c2)
            new_pop.extend([c1, c2])

        population = new_pop[:POP_SIZE]

       
        for gene in population:
            val = decode_gene(gene)
            fit = fitness(val)
            if fit > best_fit:
                best_fit, best_gene = fit, gene

        print(f"Generation {gen+1}: Best solution = {decode_gene(best_gene):.4f}, Fitness = {best_fit:.4f}")

    return decode_gene(best_gene), best_fit


best_solution, best_fitness = gene_expression_algorithm()
print("\nFinal Best Solution:", best_solution)
print("Final Best Fitness:", best_fitness)
