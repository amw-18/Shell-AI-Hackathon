import sys
sys.path.insert(1, '/path/to/Wind Farm Evaluator')

import pandas as pd 
import numpy as np 
import pyswarms as ps
import random

import matplotlib.pyplot as plt

from Vec_modified import *


def get_random_arrangement(n_turbs):

    def get_point():
        """Returns random integer from 50 to 3950 inclusive"""
        return random.uniform(50,3950)


    def is_valid(point):
        """Checks if given point is valid"""
        point = np.array(point)
        point = np.reshape(point,(1,2))
        # getting array of distances to every other point
        dist = np.linalg.norm(turbine_pos - point,axis=1)

        return min(dist) > 400   # 400 is the problem constraint

    turbine_pos = np.full((n_turbs,2),np.inf)
    count = 0
    while count < n_turbs:
        point = [get_point(),get_point()] # x,y
        if is_valid(point):
            turbine_pos[count,:] = point
            count += 1

    return turbine_pos

def get_init(n_turbs,mul=1):
    """Function to get initial valid positions for pso"""
    all_particles = np.ndarray((mul*n_turbs,2*n_turbs))
    for _ in range(mul*n_turbs):
        turb_xy = get_random_arrangement(n_turbs)
        
        ans = []
        for turb in turb_xy:
            ans.extend([turb[0],turb[1]])
            
        all_particles[_,:] = np.array(ans)

    return all_particles


def proxi_constraint(turb_coords):
    """
        Function to penalize if proximity contraint is violated.
        turb_coords is a 2d numpy array with N (xi,yi) elements.
    """
    ans = 0
    for i in range(turb_coords.shape[0]-1):
        for j in range(i+1,turb_coords.shape[0]-1):
            norm = np.linalg.norm(turb_coords[i]-turb_coords[j])
            ans += max(0,400 - norm)
            
    return ans


def obj_util(turb_coords, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a):
    """
        Objective function to be minimized w.r.t. turb_coords.
        a : the hyperparameter to adjust the proxi_val contribution
    """

    turb_coords = np.array([[turb_coords[i],turb_coords[i+1]] for i in range(0,2*n_turbs-1,2)])
    mean_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        mean_AEP += getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    
    mean_AEP /= 7
    
    proxi_val = proxi_constraint(turb_coords)
    
    ideal_AEP = 11.297*n_turbs  # 11.297 is the mean score for 1 turbine
    
    return ideal_AEP/mean_AEP + a*proxi_val - 1    # best ans is closest to zero


def obj(turb_coords, kwargs):
    ans = [None]*turb_coords.shape[0]
    for i,turb_candidate in enumerate(turb_coords):
        ans[i] = obj_util(turb_candidate, **kwargs)

    return np.array(ans)


def my_optim(n_turbs,a,c1,c2,w,kwargs):

    # hyperparameter options for optimizer c1: cognitive parameter, c2: social parameter, w: inertia
    options = {'c1' : c1, 'c2' : c2, 'w' : w}

    bounds = tuple([50*np.ones(2*n_turbs),3950*np.ones(2*n_turbs)])

    mul = 10             # multiplier for number of particles

    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=n_turbs*mul, dimensions=2*n_turbs, 
                                                    options=options, bounds=bounds, init_pos=get_init(n_turbs,mul))

    
    kwargs['n_turbs'] = n_turbs
    kwargs['a'] = a   

    
    cost, pos = optimizer.optimize(obj, iters=100, kwargs=kwargs)  # takes 5 hours with mul=10 and iter=100

    return cost, pos


if __name__ == '__main__':
    
    n_turbs = 50

    # setting turbine radius
    turb_rad = 50.0

    # Loading the power curve
    power_curve   =  loadPowerCurve('C:/Users/awals/Downloads/Shell AI Hackathon/Shell_Hackathon Dataset/power_curve.csv')

    # Loading wind data 
    years = ['07','08','09','13','14','15','17']
    wind_inst_freqs = []
    for y in years:
        wind_inst_freqs.append(binWindResourceData(f'C:/Users/awals/Downloads/Shell AI Hackathon/Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv'))
    
    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t} 

    a = 1  # not very important, just makes sure that the solution follows proximity bounds
    c2 = 0.3
    w = 0.9
    c1 = 0.5

    cost,pos = my_optim(n_turbs, a, c1, c2, w, kwargs)
    
    print('AEP is ', 11.297*n_turbs/(obj_util(pos, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a)+1))

    pos_rand = get_init(n_turbs)[0]
    print('Random AEP is', 11.297*n_turbs/(obj_util(pos_rand, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a)+1))

    pos = np.array([pos[i:i+2] for i in range(0,2*n_turbs-1,2)])


    turbines = pd.DataFrame(pos,columns=['x','y'])
    
    # uncomment following line to save results
    # turbines.to_csv("C:/Users/awals/Downloads/Shell AI Hackathon/Wind Farm Evaluator/my_trials/which_swarm_ans.csv",index=False)

    pos_rand = np.array([pos_rand[i:i+2] for i in range(0,2*n_turbs-1,2)])
    plt.scatter(pos[:,0],pos[:,1])
    plt.scatter(pos_rand[:,0],pos_rand[:,1])
    checkConstraints(pos,100.0)
    checkConstraints(pos_rand,100.0)
    plt.show()




