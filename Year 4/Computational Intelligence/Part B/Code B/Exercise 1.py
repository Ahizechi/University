from random import randint, random
import matplotlib.pyplot as plt
import numpy as np

# Create Indiviual 
def individual(min, max):
    return [randint(min, max)]

# Create the Population of Individuals
def population(count, min, max):
    return [individual(min, max) for x in range(count)]

# Define the Fitness Function. Returns the absolute value.
def fitness(individual, target):
    Total = 0
    Total = sum(individual)
    return abs(target - Total)

# Grade each target. Break the loop if the fitness has reached 0.
def grade(pop, target, fitness_history):
    sumflag = 0
    breakoutflag2 = 0
    
    for x in pop:
        if len(fitness_history) > 1:
            if fitness(x, target) == 0:
                breakoutflag2 = 1 
                sumflag = np.NaN
        else:
            sumflag = sumflag + fitness(x, target)
    return sumflag / (len(pop) * 1.0), breakoutflag2

# Create offspring. Parameters can be changed to optimise the GA. 
# Retain keeps 15% of a population to be parents and breed
# Mutate alters strings in parents
# Random Select adds individuals promoting diversity.
# Crossover breeds parents and creates children.
def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.2):
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    
    # Randomly add other individuals to promote gentic Diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # Mutate some Individuals
    for x in parents:
        if mutate > random():
            pos_to_mutate1 = randint(0, len(bin(x[0])[2:]) - 1)

            x = list(bin(x[0])[2:])
            r1 = randint(0, 1)
            x[pos_to_mutate1] = r1

            strings = [str(prnt) for prnt in x]
            x = "".join(strings)
            x = str('0b' + x)
            x = [int(x, 2)]

    # Crossover to Create Children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        
        if male != female:
            male = parents[male]
            female = parents[female]
            male = bin(male[0])[2:]
            female = bin(female[0])[2:]
            r = randint(0, len(male)-1)
            child = str('0b' + male[:r] + female[r:])
            child = [int(child, 2)]
            children.append(child)

    parents.extend(children)
    return parents


def checklist(L):
    total = 0
    for i in L:
        if isinstance(i, list): 
            total += checklist(i)
        else:
            total += i
    return total

target = 450 # Target Number
p_count = 100 # Total Number of Individuals in Population
i_min = -50 # Minimum Value of Individual
i_max = 1000 # Maximum Value of Individual
generations = 1000 # Maximum Number of Generations
p = population(p_count, i_min, i_max)

print('Initial Population: ', p)
List1 = []
fitness_history, breakoutflag2 = grade(p, target, List1)
fitness_history = [fitness_history, ]
counter = 0


# Loop through Generations and append values.
for i in range(generations):
    p = evolve(p, target)
    error1, breakoutflag2 = grade(p, target, fitness_history)
    fitness_history.append(error1)
    Results = (checklist(p) / len(p))
    print('Average Result = ', Results, ' Average Fitness: ', "{:.2f}".format(abs(target - Results)))
    counter += 1
    if breakoutflag2 == 1:
        break 

# Print Result
if breakoutflag2 == 1:
    print('Number of generations used until achieved: ', counter)

# Plot Graph
plt.plot(fitness_history, )
plt.xlabel("Generations Used")
plt.ylabel("Graded Fitness")
plt.show()