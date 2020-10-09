import random
import array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.cluster import KMeans

from deap import base
from deap import creator
from deap import tools

from sklearn.cluster import KMeans

from EA.utils import *

def initIndividual(icls, size):
    """
        Initialization function for an individual
    """
    ind = icls(get_random_arrangement(size))
    return ind

def initPopulation(pcls,icls,data_file):
    ind_list = get_arranged_location(data_file)
    pop = pcls(icls(i) for i in ind_list)
    return pop


def mainOptimize(N_TURB, IN_FILE_PATH, OUT_FILE_DIR, N_POP = 100, MAX_GEN = 100,  MODE = 'random', MultiProcess = True ,MIN_LOC = 50, MAX_LOC = 3950, CXPB = 0.65, MUTPB = 0.1, CXindpb = 0.6, MUTindpb = 0.2):
    """
        Function to implement SPEA Algorithm by calling deap.
    """
    #--------------------
    # Create Base Classes
    #--------------------
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,)) 
    creator.create("Individual", list , fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual,N_TURB)
    if MODE == 'random':
        toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=N_POP)
    elif MODE == 'readcsv':
        toolbox.register("population", initPopulation, list,creator.Individual, data_file = IN_FILE_PATH)
    toolbox.register("evaluate", evaluateAEP)

    #--------------------
    # Multiprocessing
    #--------------------
    if MultiProcess:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    #--------------------
    # Crossing and Mutation
    #--------------------
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mate", tools.cxUniform, indpb = CXindpb)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb = MUTindpb)

    toolbox.decorate("mate", checkBounds(MIN_LOC, MAX_LOC))
    toolbox.decorate("mutate", checkBounds(MIN_LOC, MAX_LOC))

    toolbox.register("selectParent", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selSPEA2)

    pop = toolbox.population()
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated Fitness of %i individuals" % len(pop))

    # Variable keeping track of the number of generations
    g = 0
    Elite = []
    fitness_lists = []
    #--------------------
    # Begin the Evolution
    #--------------------
    while g < MAX_GEN:
        g = g + 1
        print("-- Generation %i --" % g)

        # Selecting the parents for crossing
        parents = toolbox.selectParent(pop, 10)
        parents = list(toolbox.map(toolbox.clone, parents))

        top2 = tools.selBest(pop, 2)
        Elite.extend([toolbox.clone(top2[0]), toolbox.clone(top2[1])])
        if len(Elite)>5:
            Elite = tools.selBest(Elite, 5)
        fits = [ind.fitness.values[0] for ind in Elite]
        fitness_lists.append(max(fits))

        # Clone the selected individuals
        offspring = []
        # Apply crossover and mutation on the offspring
        for i in range(N_POP):
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

        # Clustering to select N_POP number of individuals 
        offspring_flat = [np.array(ind).flatten() for ind in offspring] 
        kmeans = KMeans(n_clusters = N_POP).fit(offspring_flat)
        pop = [creator.Individual(ind.reshape((50,2))) for ind in kmeans.cluster_centers_]

        #Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Re-Evaluated Fitness of %i individuals" % len(invalid_ind))
        
        # Gather all the fitnesses in one list and print the stats

        fits_pop = [ind.fitness.values[0] for ind in pop]
        
        length = len(Elite)
        mean = sum(fits) / length       
        length_pop = len(pop)
        mean_pop = sum(fits_pop) / length_pop   
        sum2 = sum(x*x for x in fits_pop)
        std_pop = abs(sum2 / length_pop - mean_pop**2)**0.5    
        
        print("  Min Eli %s" % min(fits))
        print("  Max Eli %s" % max(fits))
        print("  Max Pop %s" % max(fits_pop))
        print("  Avg Pop %s" % mean_pop)
        print("  Std Pop %s" % std_pop)
        # if g>1:
        #     print("  Del Fit %s" % fitness_lists[-1]-fitness_lists[-2])

        if g%500 == 0:
            intermediate = tools.selBest(Elite, 1)[0]
            b = pd.DataFrame(intermediate,columns=['x','y'])
            b.to_csv(OUT_FILE_DIR + "inter_data_{}.csv".format(g//500), index=False)
            f = open(OUT_FILE_DIR + "fit_vals.txt","a")
            f.write([str(intermediate.fitness.values), '\n'])
            f.close()

    top2 = tools.selBest(pop, 2)
    Elite.extend([toolbox.clone(top2[0]), toolbox.clone(top2[1])])

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(Elite, 1)[0]
    print("The Fitness of the Best individual is %s, %s" % (best_ind.fitness.values[0], float(best_ind.fitness.values[0])*574.64))

    b = pd.DataFrame(best_ind,columns=['x','y'])
    b.to_csv(OUT_FILE_DIR + "final_data.csv", index=False)

    c = pd.DataFrame(pop)
    c.to_csv(OUT_FILE_DIR + "final_pop_data.csv", index=False)

    return fitness_lists

if __name__ == '__main__':
    N_TURB = 50
    N_POP = 64
    MAX_GEN = 2

    IN_FILE_PATH = "EA/initData.csv"
    OUT_FILE_DIR = "EA/outs/"
    MODE = 'readcsv'
    mainOptimize(N_TURB = N_TURB, N_POP = N_POP, MAX_GEN = MAX_GEN, 
        IN_FILE_PATH = IN_FILE_PATH, OUT_FILE_DIR = OUT_FILE_DIR, MODE = MODE, MultiProcess = False)
