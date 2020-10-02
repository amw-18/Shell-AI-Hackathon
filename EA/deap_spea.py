import random
import array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from deap import base
from deap import creator
from deap import tools

from Wind_Farm_Evaluator.Vec_modified import *
from EA.arrangement import *


def evaluate(individual, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t):
    """
        Function to return values of both objectives as a tuple
    """

    # rearranging the terms in individual to pass to getAEP
    turb_coords = np.array([[individual[i],individual[i+1]] for i in range(0,2*n_turbs-1,2)])

    # calculating meanAEP for 
    mean_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        mean_AEP += getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    mean_AEP /= len(wind_inst_freqs)
    
    proxi_val = proxi_constraint(turb_coords)
    
    peri_val = peri_constraint(turb_coords)
    
    ideal_AEP = 11.297*n_turbs  # 11.297 is the mean score for 1 turbine
    
    return mean_AEP/ideal_AEP, proxi_val, peri_val  # First objective should be closest to 1 and second closest to zero

def peri_constraint(turb_coords):
    peri_val = 0
    for turb in turb_coords:
        for val in turb:
            if val < 50:
                peri_val += (50 - val)
            elif val > 3950:
                peri_val += (val - 3950)
                
    return peri_val

def proxi_constraint(turb_coords):
    """
        Function to penalize if proximity contraint is violated.
        turb_coords is a 2d numpy array with N (xi,yi) elements.
    """
    proxi_val = 0
    for i in range(turb_coords.shape[0]-1):
        for j in range(i+1,turb_coords.shape[0]-1):
            norm = np.linalg.norm(turb_coords[i]-turb_coords[j])
            proxi_val += max(0,400 - norm)

    return proxi_val


def initIndividual(icls, size):
    """
        Initialization function for an individual
    """
    # initializing an individual
    # ind = icls(random.uniform(FLT_MIN, FLT_MAX) for _ in range(size))
    ind = icls(get_random_arrangement(size//2))
    # initializing the individual's strategy

    return ind



N_TURB = 50

# creating a multi-objective Fitness criteria, with maximizing first and minimizing second and third
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0)) 
# creating an individual (potential solution) as a list with a fitness 
creator.create("Individual", list , fitness=creator.FitnessMulti)

# creating a toolbox
toolbox = base.Toolbox()
# registering intiES in the toolbox with an alias "individual"
toolbox.register("individual", initIndividual, creator.Individual,N_TURB)
# registering a 'population' function that will create a population(set) of n individuals (n : parameter)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

print('Done')
