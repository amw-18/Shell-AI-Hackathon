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

def initIndividual(icls, size):
    """
        Initialization function for an individual
    """
    ind = icls(get_random_arrangement(size))
    return ind


def main(N_TURB, MIN_LOC = 50, MAX_LOC = 3950, CXPB = 0.5, MUTPB = 0.01):
    #---------
    # Create
    #---------
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,)) 
    creator.create("Individual", list , fitness=creator.FitnessMulti)


    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual,N_TURB)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # register the goal / fitness function
    toolbox.register("evaluate", evaluateAEP)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)

    toolbox.decorate("mate", checkBounds(MIN_LOC, MAX_LOC))
    toolbox.decorate("mutate", checkBounds(MIN_LOC, MAX_LOC))
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selSPEA2)


    pop = toolbox.population(n=300)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 1 and g < 100:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

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
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length       
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
    
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))



if __name__ == '__main__':
    N_TURB = 50
    main(N_TURB)
