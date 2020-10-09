import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.optimize import Bounds

from Wind_Farm_Evaluator.Vec_modified import *



def objective(curr_particle, kwargs):
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
        
        mean_AEP = aggr_AEP/len(kwargs['wind_inst_freqs'])  # negative because we want to maximize AEP
        

        return (mean_AEP)    #(mean_AEP/kwargs['ideal_AEP'])


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



def proxi_constraint(particle):
    """
        Function to penalize if proximity contraint is violated.
        particle : numpy array with shape (n_turbs, 2)
    """
    particle = particle.reshape((50, 2))
    proxi_penalty = 0

    for i in range(particle.shape[0]-1):  
        for j in range(i+1,particle.shape[0]):
            norm = np.linalg.norm(particle[i]-particle[j])
            proxi_penalty += norm-400  # linear penalty 
            
    return proxi_penalty   # dividing to normalize the value to between 0 and 1               





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





def SLS_main(n_turbs):

    ineq_cons = {'type': 'ineq',
                'fun' : proxi_constraint }

    
    x0 = np.array(pd.read_csv(r'C:\Users\Avinash\Documents\SHELL\CODE\Shell-AI-Hackathon\Trials\opt_swarm_ans15.csv')).flatten() 
    #x0 = get_random_arrangement(n_turbs)    
    #x0 = np.random.randint(50,3950,100) 
    bounds = Bounds(50*np.ones(2*n_turbs), 3950*np.ones(2*n_turbs))
    
    kwargs = parse_data_PSO(n_turbs)
    
    print(objective(x0,kwargs))

    sol = minimize(objective, x0, args=kwargs, method='SLSQP', options ={'disp' :True}, tol=1e-10,constraints=[ineq_cons], bounds=bounds)    

    return sol  


    