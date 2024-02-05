import numpy as np

TARGET_STRING = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

def fitness(individual):
    return -sum(np.abs(TARGET_STRING - individual))

def population_fitness(population):
    return np.mean([fitness(individual) for individual in population])

def mutate(individual):
    mutations = np.random.rand(len(individual)) < 0.01
    individual[mutations] = 1 - individual[mutations]
    return individual

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1,30)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

if __name__ == '__main__':
    population = np.random.randint(2, size=(100,30))
    print(f"Fitness at generation 0: {population_fitness(population)}")

    for generation in range(50):
        fitness_score = np.array([fitness(individual) for individual in population])
        indices = np.argsort(fitness_score)[::-1]
        fittest = population[indices[:50]]

        offspring = []
        for i in range(len(fittest)):
            parent1, parent2 = fittest[np.random.choice(len(fittest), size=2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])

        population = np.array(offspring)
        print(f"Fitness at generation {generation+1}: {population_fitness(population)}")
