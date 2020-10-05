import random
import array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing

from deap import base
from deap import creator
from deap import tools

from EA.utils import *

def initIndividual(icls, size, mode = 'random' ):
    """
        Initialization function for an individual
    """
    if mode=='random':
        ind = icls(get_random_arrangement(size))
    else:
        ind = icls(size)
    return ind

def initPopulation(pcls,icls):
    ind_list = get_arranged_location()
    pop = pcls(icls(i) for i in ind_list)
    return pop


def main(N_TURB, N_POP = 100, MIN_LOC = 50, MAX_LOC = 3950, CXPB = 0.5, MUTPB = 0.2):
    #---------
    # Create
    #---------
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,)) 
    creator.create("Individual", list , fitness=creator.FitnessMulti)


    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual,N_TURB)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("population", initPopulation, list,creator.Individual)

    # register the goal / fitness function
    toolbox.register("evaluate", evaluateAEP)

    pool = multiprocessing.Pool()
    toolbox.register("map", map)
    # register the crossover operator
    toolbox.register("mate", tools.cxUniform, indpb = 0.5)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)

    toolbox.decorate("mate", checkBounds(MIN_LOC, MAX_LOC))
    toolbox.decorate("mutate", checkBounds(MIN_LOC, MAX_LOC))
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("selectParent", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selSPEA2)


    pop = toolbox.population(n=N_POP)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    Elite = []
    # Begin the evolution
    while max(fits) < 1 and g < 100:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        # offspring = toolbox.select(pop, len(pop))
        parents = toolbox.selectParent(pop, 10)
        parents = list(toolbox.map(toolbox.clone, parents))
        # parents = tools.selBest(pop, 10)
        top2 = tools.selBest(pop, 2)
        Elite.extend([toolbox.clone(top2[0]), toolbox.clone(top2[1])])
        if len(Elite)>5:
            Elite = tools.selBest(Elite, 5)

        # Clone the selected individuals
        # offspring = list(toolbox.map(toolbox.clone, offspring))
        offspring = []
        # Apply crossover and mutation on the offspring
        for i in range(len(pop)//2):
            child1, child2 = random.choices([*parents, *Elite],k=2)
            # cross two individuals with probability CXPB
            offspring.extend([toolbox.clone(child1), toolbox.clone(child2)])
            if random.random() < CXPB:
                toolbox.mate(offspring[-1], offspring[-2])
                # fitness values of the children
                # must be recalculated later
                del offspring[-1].fitness.values
                del offspring[-2].fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # for child1, child2 in zip(offspring[::2], offspring[1::2]):

        #     # cross two individuals with probability CXPB
        #     if random.random() < CXPB:
        #         toolbox.mate(child1, child2)

        #         # fitness values of the children
        #         # must be recalculated later
        #         del child1.fitness.values
        #         del child2.fitness.values

        # for mutant in offspring:

        #     # mutate an individual with probability MUTPB
        #     if random.random() < MUTPB:
        #         toolbox.mutate(mutant)
        #         del mutant.fitness.values

        # #Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in Elite]
        
        length = len(Elite)
        mean = sum(fits) / length       
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
    
    top2 = tools.selBest(pop, 2)
    Elite.extend([toolbox.clone(top2[0]), toolbox.clone(top2[1])])

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(Elite, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    b = pd.DataFrame(best_ind,columns=['x','y'])
    b.to_csv("EA/data_rand.csv", index=False)

    c = pd.DataFrame(pop)
    c.to_csv("EA/Pop_data_rand.csv", index=False)

if __name__ == '__main__':
    N_TURB = 50
    N_POP = 64
    main(N_TURB,N_POP)
