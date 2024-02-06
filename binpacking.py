import matplotlib.pyplot as plt
import numpy as np

GENERATIONS = 100
POPULATION = 1000

def generate_population():
    population = []
    for _ in range(POPULATION):
        individual = [[] for _ in range(NUM_ITEMS)]
        for item in ITEMS:
            random_index = np.random.choice(NUM_ITEMS)
            individual[random_index].append(item)
        population.append(individual)
    return np.array(population)

def num_bins_used(individual):
    used = 0
    for bin in individual:
        if np.sum(bin) > 0:
            used += 1
    return used

def fitness(individual):

    # Check all items exist
    items_omitted = 0
    to_check = ITEMS.copy()
    for bin in individual:
        for item in bin:
            if item not in to_check:
                items_omitted +=1
            else:
                to_check.remove(item)
    items_omitted += len(to_check)

    # Calculate number of bins underflowed
    bins_underflowed = 0
    for bin in individual:
        if np.sum(bin) > 0 and np.sum(bin) < BIN_CAPACITY:
            bins_underflowed += BIN_CAPACITY - np.sum(bin)

    # Calculate number of bins overflowed
    bins_overflowed = 0
    for bin in individual:
        if np.sum(bin) > 0 and np.sum(bin) > BIN_CAPACITY:
            bins_overflowed += np.sum(bin) - BIN_CAPACITY


    return -bins_overflowed-bins_underflowed-(items_omitted*max(ITEMS))

def population_fitness(population):
    return np.mean([fitness(individual) for individual in population])

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual):
    for bin in individual:
        mutate = np.random.rand() < 0.000001
        if mutate and len(bin) > 0:
            move_item = bin.pop(np.random.randint(len(bin)))
            new_bin = np.random.randint(len(individual))
            individual[new_bin].append(move_item)
    return individual

def genetic_algorithm_binpacking():
    population = generate_population()
    generational_fitness = [population_fitness(population)]
    print(f"Fitness at generation 0: {population_fitness(population)}")

    for generation in range(GENERATIONS):
        fitness_score = np.array([fitness(individual) for individual in population])
        indices = np.argsort(fitness_score)[::-1]
        fittest = population[indices[:POPULATION//2]]

        offspring = []
        for i in range(len(fittest)):
            parent1, parent2 = fittest[np.random.choice(len(fittest), size=2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])

        population = np.array(offspring)
        generational_fitness.append(population_fitness(population))
        print(f"Fitness at generation {generation+1}: {population_fitness(population)}, Bins: {num_bins_used(population[np.argmax(fitness_score)])}")

    np.set_printoptions(threshold=np.inf)
    fittest_individual = population[np.argmax(fitness_score)]
    print(fittest)
    bins_used = num_bins_used(fittest_individual)
    print(f"Number of bins to use: {bins_used}")
    plt.plot(range(GENERATIONS+1), generational_fitness, marker="o")
    plt.title("Fitness by Generation")
    plt.xlabel("Generation")
    plt.ylabel("Mean Population Fitness")
    plt.show()

if __name__ == '__main__':
    global ITEMS, BIN_CAPACITY, NUM_ITEMS
    for i in range(3,6):
        content = open(f"binpacking_tests/sample{i}.txt").read().split("\n")
        BIN_CAPACITY = int(content[2])
        ITEMS = []
        for line in content[3:]:
            number, times = [int(x) for x in line.split(" ")]
            [ITEMS.append(number) for _ in range(times)]
        NUM_ITEMS = len(ITEMS)
        print(ITEMS)
        genetic_algorithm_binpacking()