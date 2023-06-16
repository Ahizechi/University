from random import randint, random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Define the Target Curve
def targetcurve(x):
    return 25 * x ** 5 + 18 * x ** 4 + 31 * x ** 2 - 14 * x ** 2 + 7 * x -19

# Create Indiviual 
def individual(min, max):
    return [randint(min, max) for x in range(length)]

# Create the Population of Individuals
def population(count, min, max):
    return [individual(min, max) for x in range(count)]

# Define the Fitness Function. Returns the absolute value.
def fitness(individual, target):
    target = np.array(target)
    individual = np.array(individual)
    fit = target - individual
    return np.average(abs(fit))

# Grade each target. Break the loop if the fitness has reached 0.
def grade(pop, target):
    summed = 0
    breakoutflag = 0
    for x in pop:
        if fitness(x, target) == 0:
            breakoutflag = 1
            summed = np.NaN
            break
        else:
            summed = summed + fitness(x, target)
    return summed / (len(pop) * 1.0), breakoutflag

# Create offspring. Parameters can be changed to optimise the GA.
def evolve(pop, target, retain=0.15, random_select=0.05, mutate=0.3):
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)] 
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            individual[pos_to_mutate] = randint(i_min, i_max)

    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []

    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)

        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents


target = [25, 18, 31, -14, 7, -19] # Target List
p_count = 100 # Total Number of Individuals in Population
i_min = -50 # Minimum Value of Individual
i_max = 50 # Maximum Value of Individual
length = 6 # Length of Individuals
generations = 250 # Maximum Number of Generations

p = population(p_count, i_min, i_max)
fitness_history, breakoutflag = grade(p, target)
fitness_history = [fitness_history, ]

list1 = []
list2 = []

# Target Curve within range -100 to 100.
for x in range(-100, 100, 10):
    y = targetcurve(x)
    list1.append(y)
    list2.append(x)

# Loop until Fitness reaches 0 or Generation cap is reached. TQDM shows progress bar.
for i in tqdm(range(generations)):
    p = evolve(p, target)
    average_error, breakoutflag = grade(p, target)
    fitness_history.append(average_error)
    if breakoutflag == 1:
        break

graded = [(fitness(x, target), x) for x in p]
graded = [x[1] for x in sorted(graded)]
parents = graded[0]
print(parents)

list3 = []
list4 = []

# Plot GA coefficient polynomial within range of -100 to 100.
for x in range(-100, 100, 10):
    y = parents[0] * x ** 5 + parents[1] * x ** 4 + parents[2] * x ** 3 + parents[3] * x ** 2 +  parents[4] * x - parents[5]
    list3.append(x)
    list4.append(y)
    
# Plot Graph of Results
plt.subplot(121)
plt.plot(list2, list1, label = "Target Curve")
plt.plot(list3, list4, '--', label = "Output Curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.subplot(122)
plot2 = plt.plot(fitness_history)
plt.xlabel("Generations")
plt.ylabel("Population Fitness")
plt.show()