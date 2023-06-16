import matplotlib.pyplot as plt

from functools import reduce
from operator import add
from random import randint, random

# Individual Created between Min and Max. Called p_count times to create population.
def individual(length, min, max):
    return [ randint(min,max) for x in range(length) ] 

# Create Population of Inidividuals
def population(count, length, min, max):
    return [ individual(length, min, max) for x in range(count) ]

# Calculate Fitness of Individuals
def fitness(individual, target):
    sum = reduce(add, individual, 0)
    return abs(target-sum)

# Ranking each Inidividual
def grade(pop, target):
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)

# Create offspring. Parameters can be changed to optimise the GA.
def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)

            individual[pos_to_mutate] = randint(
                min(individual), max(individual))

    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)

        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)     

    parents.extend(children)
    return parents


target = 550 # Target Number
p_count = 100 # Total Number of Individuals in Population
i_length = 6 # Length of Individuals
i_min = 1 # Minimum Value of Individual
i_max = 1000 # Maximum Value of Individual
generations = 1000 # Maximum Number of Generations
p = population( p_count, i_length, i_min, i_max )
fitness_history = [grade(p, target)]

breakoutflag = False

# Loop until Fitness reaches 0, fitness has repeated itself, or Generation cap is reached.
for i in range(generations):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))
    for datum in fitness_history:
        if datum == 0:
            breakoutflag = True
            print('Number of Iterations:', len(fitness_history))
            break
        if len(fitness_history) > 1 and datum != 0:
            while fitness_history[-1] == fitness_history[-5:]:
                breakoutflag = True
                break
    if breakoutflag:
        break

print('Target Number:', target)
print('Population Size:', p_count)
print('Maximum Number of Generations:', generations)
print('Final Fitness Value:', fitness_history[-1])
plt.plot(fitness_history)
plt.xlabel("Generations")
plt.ylabel("Fitness History")
plt.title('Target Number: %3.2f' % target)
plt.grid()
plt.show()


