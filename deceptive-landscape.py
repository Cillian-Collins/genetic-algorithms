import matplotlib.pyplot as plt
import numpy as np

def fitness(individual):
    individual_fitness = sum(individual)
    if individual_fitness != 0:
        return individual_fitness
    ret = 2*len(individual)
    print(ret)
    input("continue?")
    return ret

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
    generational_fitness = [population_fitness(population)]
    print(f"Fitness at generation 0: {population_fitness(population)}")

    for generation in range(1000):
        fitness_score = np.array([fitness(individual) for individual in population])
        indices = np.argsort(fitness_score)[::-1]
        fittest = population[indices[:len(population)//2]]

        offspring = []
        for i in range(len(fittest)):
            parent1, parent2 = fittest[np.random.choice(len(fittest), size=2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])

        population = np.array(offspring)
        generational_fitness.append(population_fitness(population))
        print(f"Fitness at generation {generation+1}: {population_fitness(population)}")
    plt.plot(range(1001), generational_fitness, marker="o")
    plt.title("Fitness by Generation")
    plt.xlabel("Generation")
    plt.ylabel("Mean Population Fitness")
    plt.show()