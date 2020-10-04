import numpy as np
import pandas as pd
import pyswarms as ps 
from pyswarms.utils.plotters import plot_cost_history

import matplotlib.pyplot as plt 

from Wind_Farm_Evaluator.Vec_modified import *


def get_random_arrangement(n_turbs):
    """
    Gets a valid random individual as a numpy array of shape (n_turbs, 2)
    """

    def get_turb(a, b):
        """ 
            Returns two random numbers between a and b from a unifrom distribution
            as a numpy array
        """
        return np.random.uniform(a, b, 2)

    def is_valid(individual, turb):
        """
            Checks if the turbine is a valid fit inside the individual
            individual : numpy array of turbines with shape (n_turbs, 2)
            turb : one turbine with shape (1, 2)
        """
        distances = np.linalg.norm(individual - turb, axis=1)
        return min(distances) > 400

    rand_ind = np.full((n_turbs, 2), np.inf)
    count = 0
    while count < n_turbs:
        turb = get_turb(50, 3950)
        if is_valid(rand_ind, turb):
            rand_ind[count,:] = turb
            count += 1

    return rand_ind


def get_init(n_turbs, n_part=1):
    """
        Function to get initial valid positions for n_part particles
    """
    all_particles = np.ndarray((n_part,2*n_turbs))
    for _ in range(n_part):
        particle = get_random_arrangement(n_turbs)
        all_particles[_,:] = particle.flatten()

    return all_particles


def proxi_constraint(particle):
    """
        Function to penalize if proximity contraint is violated.
        particle : numpy array with shape (n_turbs, 2)
    """
    proxi_penalty = 0
    for i in range(particle.shape[0]-1):  
        for j in range(i+1,particle.shape[0]):
            norm = np.linalg.norm(particle[i]-particle[j])
            proxi_penalty += max(0, 401-norm)  # linear penalty 
            
    return proxi_penalty/(particle.shape[0]*400)    # dividing to normalize the value to between 0 and 1


def obj(swarm, kwargs):
    """
        Returns value of objective function for each particle in the swarm as a 1-D
        numpy array of shape (n_particles,)
    """
    def obj_util(curr_particle, kwargs):
        """
            Objective function to be minimized w.r.t. particle.
            a : weight to use for proxi_penalty
        """
        particle = curr_particle.reshape((kwargs['n_turbs'], 2))
        aggr_AEP = 0
        for wind_inst_f in kwargs['wind_inst_freqs']:
            aggr_AEP += getAEP(kwargs['turb_rad'], particle, kwargs['power_curve'], wind_inst_f,
                                kwargs['n_wind_instances'], kwargs['cos_dir'], kwargs['sin_dir'], 
                                kwargs['wind_sped_stacked'], kwargs['C_t'])
        
        mean_AEP = -aggr_AEP/len(kwargs['wind_inst_freqs'])  # negative because we want to maximize AEP
        
        proxi_penalty = proxi_constraint(particle)
        
        return mean_AEP/kwargs['ideal_AEP'] + kwargs['a']*proxi_penalty

    obj_vals = np.ndarray((swarm.shape[0],))
    for i, particle in enumerate(swarm):
        obj_vals[i] = obj_util(particle, kwargs)

    return obj_vals


def get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=None, v_clamp=False):
    """
        Get optimizer with given values
        v_clamp: (False) setting True will set clamps to (-800, 800)
        init_vals: (None - 'random arrangement') starting points
    """
    options = {'c1': c1, 'c2': c2, 'w': w}
    bounds = tuple([50*np.ones(2*n_turbs), 3950*np.ones(2*n_turbs)])
    if v_clamp:
        v_clamp = (-800, 800)
    else:
        v_clamp = (-np.inf, np.inf)

    if init_vals is None:
        init_vals = get_init(n_turbs, n_part)
  
    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=n_part, dimensions=2*n_turbs, ftol=1e-08, ftol_iter=15,
                            velocity_clamp=v_clamp, options=options, bounds=bounds, init_pos=init_vals)

    return optimizer


def parse_data_PSO(n_turbs):
    """
        Get data from csv(s) for calculating AEP
    """

    # setting turbine radius
    turb_rad = 50.0

    # Loading the power curve
    power_curve   =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')

    # Loading wind data 
    years = ['07']#,'08','09','13','14','15','17']
    wind_inst_freqs = []
    for y in years:
        wind_inst_freqs.append(binWindResourceData(f'./Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv'))
    
    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    # getting ideal_AEP for particular wind instances and n_turbs
    ref_loc = get_random_arrangement(1)
    ideal_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        ideal_AEP += getAEP(turb_rad, ref_loc, power_curve, wind_inst_freq, *preProcessing(power_curve, 1))
    ideal_AEP /= len(wind_inst_freqs)
    ideal_AEP *= n_turbs

    # creating a dictionary for extracted values
    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs, 'n_turbs': n_turbs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t, 'ideal_AEP': ideal_AEP, 'a': 100}

    return kwargs