import sys
sys.path.insert(1, '/path/to/Wind Farm Evaluator')


import numpy as np
import pandas as pd
import pyswarms as ps 

import matplotlib.pyplot as plt 

from Vec_modified import *

def get_random_arrangement(n_turbs):
    """
    Gets a valid random individual as a numpy array of shape (n_turbs, 2)
    """

    def get_turb(a, b):
        """ 
            returns two random numbers between a and b from a unifrom distribution
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
            proxi_penalty += max(0, 400-norm)  # linear penalty ##max(0, 400-norm)
            
    return proxi_penalty/(particle.shape[0]*400)    # dividing to normalize the value between 0 and 1
        
def obj_util(curr_particle, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a):
    """
        Objective function to be minimized w.r.t. particle.
        a : weight to use for proxi_penalty
    """
    global ideal_AEP
    particle = curr_particle.reshape((n_turbs, 2))
    aggr_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        aggr_AEP += getAEP(turb_rad, particle, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    
    mean_AEP = -aggr_AEP/len(wind_inst_freqs)  # negative because we want to maximize AEP
    
    proxi_penalty = proxi_constraint(particle)
    
    return mean_AEP/ideal_AEP #+ a*proxi_penalty

def obj(swarm, kwargs):
    """
        Returns value of objective function for each particle in the swarm as a 1-D numpy
        array of shape (n_particles,)
    """
    ans = np.ndarray((swarm.shape[0],))
    for i, particle in enumerate(swarm):
        ans[i] = obj_util(particle, **kwargs)

    return ans

def my_optim(n_turbs,a,c1,c2,w,kwargs):

    # hyperparameter options for optimizer c1: cognitive parameter, c2: social parameter, w: inertia
    options = {'c1' : c1, 'c2' : c2, 'w' : w}

    bounds = tuple([50*np.ones(2*n_turbs),3950*np.ones(2*n_turbs)])
    v_clamp = (-800, 800)

    n_part = 64    # number of particles in the swarm (20-70) suggested

    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=n_part, dimensions=2*n_turbs, ftol=1e-12, ftol_iter=10, center=[2000]*2*n_turbs,
                            velocity_clamp=v_clamp, options=options, bounds=bounds, init_pos=get_init(n_turbs, n_part))

    
    kwargs['n_turbs'] = n_turbs
    kwargs['a'] = a    

    return optimizer.optimize(obj, iters=200, kwargs=kwargs)


if __name__ == '__main__':
    
    n_turbs = 10

    # setting turbine radius
    turb_rad = 50.0

    # Loading the power curve
    power_curve   =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')

    # Loading wind data 
    years = ['07']
    wind_inst_freqs = []
    for y in years:
        wind_inst_freqs.append(binWindResourceData(f'./Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv'))
    
    wind_inst_freqs_ = []
    for wind_inst_freq in wind_inst_freqs:
        mod = wind_inst_freq.reshape((36, 15))
        for row in range(36):
            mod[row, :4] = [0]*4
        mod = mod/np.sum(mod)
        wind_inst_freqs_.append(mod.ravel())

    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs_,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t} 

    # getting ideal_AEP for normalization purpose
    ref_loc = get_random_arrangement(1)
    ideal_AEP = 0
    for wind_inst_freq_ in wind_inst_freqs_:
        ideal_AEP += getAEP(turb_rad, ref_loc, power_curve, wind_inst_freq_, *preProcessing(power_curve, 1))
    
    ideal_AEP /= len(wind_inst_freqs)
    ideal_AEP *= n_turbs

    # defining parameters for optimization
    a = 100  # weight for the proximity penalty -- critical only if random initialization done
    c2 = 1.49  # social
    w = 0.4  # inertia
    c1 = 1.49  # cognitive

    cost,pos = my_optim(n_turbs, a, c1, c2, w, kwargs)
    print('AEP is ', -ideal_AEP*obj_util(pos, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, 0))
    pos = pos.reshape((n_turbs, 2))
    checkConstraints(pos,100.0)

    pos_rand = get_random_arrangement(n_turbs).flatten()
    print('Random AEP is', -ideal_AEP*obj_util(pos_rand, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, 0))
    pos_rand = pos_rand.reshape((n_turbs, 2))
    checkConstraints(pos_rand,100.0)

    
    ### uncomment following line to save results ###
    # turbines = pd.DataFrame(pos,columns=['x','y'])
    # turbines.to_csv("C:/Users/awals/Downloads/Shell AI Hackathon/Wind Farm Evaluator/my_trials/which_swarm_ans.csv",index=False)

    plt.scatter(pos[:,0],pos[:,1])
    plt.scatter(pos_rand[:,0],pos_rand[:,1])
    plt.show()