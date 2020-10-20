import numpy as np
import pandas as pd
import pyswarms as ps 
from pyswarms.utils.plotters import plot_cost_history

import matplotlib.pyplot as plt 

from Wind_Farm_Evaluator.Vec_modified import *


def get_random_arrangement(n_turbs, a=450, b=3550):
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
        turb = get_turb(a, b)
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
            proxi_penalty += max(0, 400-norm)  # linear penalty 
            
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
        particle = np.append(kwargs['bord_turbs'].flatten(), curr_particle)
        particle = particle.reshape((kwargs["n_turbs"], 2))
        aggr_AEP = 0
        for wind_inst_f in kwargs['wind_inst_freqs']:
            aggr_AEP += getAEP(kwargs['turb_rad'], particle, kwargs['power_curve'], wind_inst_f,
                                kwargs['n_wind_instances'], kwargs['cos_dir'], kwargs['sin_dir'], 
                                kwargs['wind_sped_stacked'], kwargs['C_t'])
        
        mean_AEP = -aggr_AEP/len(kwargs['wind_inst_freqs'])  # negative because we want to maximize AEP
        
        # proxi_penalty = proxi_constraint(particle)
        

        return mean_AEP/kwargs['ideal_AEP'] #+ kwargs['a']*proxi_penalty

    obj_vals = np.ndarray((swarm.shape[0],))
    for i, particle in enumerate(swarm):
        obj_vals[i] = obj_util(particle, kwargs)

    return obj_vals


def get_optimizer(n_part, c1, c2, w, n_turbs, init_vals):
    """
        Get optimizer with given values 
        v_clamp: (False) setting True will set clamps to (-800, 800)
        init_vals: (None - 'random arrangement') starting points
    """
    in_turbs = 1
    options = {'c1': c1, 'c2': c2, 'w': w}
    bounds = tuple([450*np.ones(2*in_turbs), 3550*np.ones(2*in_turbs)])
  
    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=n_part, dimensions=2*in_turbs, ftol=1e-08, ftol_iter=15,
                             options=options, bounds=bounds, init_pos=init_vals, bh_strategy='my_strategy')

    return optimizer


def parse_data_PSO(n_turbs, years, ignore=None):
    """
        Get data from csv(s) for calculating AEP
        years: years to use for optimizing
        ignore: number to remove wind data upto that wind speed bin
                (default is None)
    """

    # setting turbine radius
    turb_rad = 50.0

    # Loading the power curve
    power_curve   =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')

    # Loading wind data 
    wind_inst_freqs = []
    all_wind = np.zeros((540,))
    for y in years:
        all_wind += binWindResourceData(f'Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv')
    wind_inst_freqs.append(all_wind/np.sum(all_wind))
     
    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    # getting ideal_AEP for particular wind instances and n_turbs
    ref_loc = get_random_arrangement(1)
    ideal_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        ideal_AEP += getAEP(turb_rad, ref_loc, power_curve, wind_inst_freq, *preProcessing(power_curve, 1))
    ideal_AEP /= len(wind_inst_freqs)
    ideal_AEP *= n_turbs
    
    if ignore is not None:
        for wind_inst_freq in wind_inst_freqs:
            wind_inst_freq = wind_inst_freq.reshape((36, 15))
            wind_inst_freq[:,:ignore] = 0
            wind_inst_freq = wind_inst_freq/np.sum(wind_inst_freq)

    bord_turbs = get_border_arrangement()
    # creating a dictionary for extracted values
    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs, 'n_turbs': n_turbs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t, 'ideal_AEP': ideal_AEP, 'a': 100, 'bord_turbs': bord_turbs}

    return kwargs


def get_border_arrangement():
    # n_bord = [8]*4
    # bord_vals = [np.linspace(50, 3950, n_bord[i]+2) for i in range(4)]
    # left_bound = [np.array([50, val]) for val in bord_vals[0][1:-1]]
    # top_bound = [np.array([val, 3950]) for val in bord_vals[1]]
    # right_bound = [np.array([3950, val]) for val in bord_vals[2][1:-1]]
    # bottom_bound = [np.array([val, 50]) for val in bord_vals[3]]
    # ans = [*top_bound, *right_bound, *left_bound, *bottom_bound]
    # ans = np.array(ans)
    ans = np.array(pd.read_csv('C:/Users/awals/Downloads/Shell AI Hackathon/PSO/brute7/43.csv'))
    # plt.scatter(ans[:,0],ans[:,1])
    # plt.show()
    return ans

