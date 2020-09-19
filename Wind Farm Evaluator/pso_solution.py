import sys
sys.path.insert(1, '/path/to/Wind Farm Evaluator')

import numpy as np 
import pyswarms as ps
import random

import matplotlib.pyplot as plt

from Vec_modified import *


def proxi_constraint(turb_coords):
    """
        Function to penalize if proximity contraint is violated.
        turb_coords is a 2d numpy array with N (xi,yi) elements.
    """
    ans = 0
    for i in range(turb_coords.shape[0]-1):
        for j in range(i+1,turb_coords.shape[0]-1):
            norm = np.linalg.norm(turb_coords[i]-turb_coords[j])
            if norm < 400:
                ans += (4000/norm)**2
            
    return ans


def obj_util(turb_coords, n_turbs, turb_rad, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a):
    """
        Objective function to be minimized w.r.t. turb_coords.
        a : the hyperparameter to adjust the proxi_val contribution
    """

    turb_coords = np.array([[turb_coords[i],turb_coords[i+1]] for i in range(0,2*n_turbs-1,2)])
    AEP = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

    proxi_val = proxi_constraint(turb_coords)

    return 3000/AEP + a*proxi_val


def obj(turb_coords, kwargs):
    ans = [None]*turb_coords.shape[0]
    for i,turb_candidate in enumerate(turb_coords):
        ans[i] = obj_util(turb_candidate, **kwargs)

    return np.array(ans)


def my_optim(n_turbs,a,c1,c2,w,kwargs):

    # hyperparameter options for optimizer c1: cognitive parameter, c2: social parameter, w: inertia
    options = {'c1' : c1, 'c2' : c2, 'w' : w}

    bounds = tuple([50*np.ones(2*n_turbs),3950*np.ones(2*n_turbs)])

    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=n_turbs*10, dimensions=2*n_turbs, options=options, bounds=bounds)

    
    kwargs['n_turbs'] = n_turbs
    kwargs['a'] = a   

    
    cost, pos = optimizer.optimize(obj, iters=100, kwargs=kwargs)

    return cost, pos


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

if __name__ == '__main__':
    n_turbs = 15
    a = 1
    c1 = 0.5
    c2 = 0.3
    w = 0.9
    
    # setting turbine radius
    turb_rad = 50.0
    # Loading the power curve
    power_curve   =  loadPowerCurve('Shell_Hackathon Dataset/power_curve.csv')
    # Loading wind data 
    wind_inst_freq =  binWindResourceData('Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freq': wind_inst_freq,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t} 

    cost, pos = my_optim(n_turbs,a,c1,c2,w,kwargs)

    pos = np.array([pos[i:i+2] for i in range(0,2*n_turbs-1,2)])
    checkConstraints(pos,100.0)
    print('AEP is ', getAEP(turb_rad, pos, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))

    pos_rand = get_random_arrangement(n_turbs)
    print('Random AEP is', getAEP(turb_rad, pos_rand, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
    
    plt.scatter(pos[:,0],pos[:,1])
    plt.scatter(pos_rand[:,0],pos_rand[:,1])
    plt.show()




