import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing

from deap import base
from deap import creator
from deap import tools

from EA.jade_utils import *


def init_individual(icls, size, is_smart):
    """
        Initialization function for an individual
    """
    if is_smart:
        func = get_smart_arrangement
    else:
        func = get_random_arrangement
        
    ind = icls(func(size))
    return ind


# def update_ind(individual, pop, mu1, mu2, n_jrand):
#     """
#         updates the individual
#         pop: list of individuals in population sorted according to fitness values
#         returns an updated copy of the individual and parameter values that resulted
#         in successful update
#     """
#     global toolbox
#     N = len(individual)

#     # selecting the best
#     s_best = pop[-1]

#     # selecting a unique random individual
#     while 1:
#         s1 = random.choice(pop[:-1])
#         if s1 is not individual:
#             break

#     # creating a distorted individual out of the selected one
#     s2 = np.array(random.sample(list(s1), N))

#     # getting -1 or 1 randomly
#     sgn = np.sign(random.uniform(-1, 1))

#     # getting the parameters to use
#     F1, F2 = random.normalvariate(mu1, 1), random.normalvariate(mu2, 1)

#     # computing the velocity of the particle
#     D = sgn*(F1*s1 - F2*s2)
#     v = s_best + D

#     # getting the number of turbs to cross
#     ni_jrand = int(random.normalvariate(n_jrand, 1))
#     ni_jrand = max(1,min(N,n_jrand))

#     # crossing the turbs
#     updated_individual = [*random.sample(list(v), ni_jrand), *random.sample(list(individual), N-ni_jrand)]

#     # repairing the updated individual inplace if contraints violated
#     repair(updated_individual)

#     # creating "Individual" object of the updated+repaired values
#     updated_individual = creator.Individual(updated_individual)

#     # evaluating and providing fitness value to the updated individual
#     toolbox.evaluate(updated_individual)

#     # selecting the better individual
#     if individual.fitness.dominates(updated_individual.fitness):
#         return individual, None, None, None
#     else:
#         return updated_individual, F1, F2, ni_jrand
    


# def mainOptimize(n_turbs, pop_size, is_smart, MultiProcess=False):
#     """
#         Function to implement SPEA Algorithm by calling deap.
#     """

    # years = ['09']#,'08','09','13','14','15','17']
    # kwargs = parse_data(n_turbs, years)
    # #--------------------
    # # Create Base Classes
    # #--------------------
    # creator.create("Fitness", base.Fitness, weights=(1.0,)) 
    # creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    # toolbox = base.Toolbox()
    # toolbox.register("individual", init_individual, creator.Individual, n_turbs, is_smart)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)
    # toolbox.register("evaluate", evaluateAEP, data=kwargs)

    # #--------------------
    # # Multiprocessing
    # #--------------------
    # if MultiProcess:
    #     pool = multiprocessing.Pool()
    #     toolbox.register("map", pool.map)
    # else:
    #     toolbox.register("map", map)

    # toolbox.register("update", update_ind)

    # mu1, mu2, n_jrand = 1, 1, 5
    # pop = toolbox.population()
    # list(toolbox.map(toolbox.evaluate, pop))
    # # sorting the population so that best individual is last
    # pop.sort(key=lambda x: x.fitness.values[0])
    # print(f"---Generation {0}---")
    # print("Population mean...", pop[0].fitness.values[0])
    # for gen in range(1000):
    #     print(f"---Generation {gen+1}---")

    #     # mutation and cross operation
    #     ans = list(toolbox.map(update_ind, pop, [pop]*pop_size,
    #          [mu1]*pop_size, [mu2]*pop_size, [n_jrand]*pop_size))

    #     # updating population
    #     pop = [ind[0] for ind in ans]

    #     # getting successful values of parameters
    #     S_f1 = [ind[1] for ind in ans if ind[1] is not None]
    #     S_f2 = [ind[2] for ind in ans if ind[2] is not None]
    #     S_n_jrand = [ind[3] for ind in ans if ind[3] is not None]
        
    #     # updating parameter values
    #     mu1 = np.mean(S_f1)
    #     mu2 = np.mean(S_f2)
    #     n_jrand = np.mean(n_jrand)

    #     # sorting the population so that best individual is last
    #     pop.sort(key=lambda x: x.fitness.values[0])
    #     print("Population best..", pop[0].fitness.values[0])
    

# if __name__ == '__main__':
#     N_TURB = 50
#     N_POP = 4
#     MAX_GEN = 2

#     IN_FILE_PATH = "EA/initData.csv"
#     OUT_FILE_DIR = "EA/outs/"
#     MODE = 'readcsv'
#     mainOptimize(N_TURB = N_TURB, N_POP = N_POP, MAX_GEN = MAX_GEN, 
#         IN_FILE_PATH = IN_FILE_PATH, OUT_FILE_DIR = OUT_FILE_DIR, MODE = MODE, MultiProcess = False)
