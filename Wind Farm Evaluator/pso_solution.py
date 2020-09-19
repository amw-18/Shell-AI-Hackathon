import sys
sys.path.insert(1, '/path/to/Wind Farm Evaluator')

import numpy as np 
import pyswarms as ps

import matplotlib.pyplot as plt

from Farm_Evaluator_Vec import *

# def peri_constraint(turb_coords):
#     """
#         Function to penalize if perimeter constraint is violated.
#         turb_coords is an array with [x1,y1,x2....xN,yN] (2N elements)
#     """
#     ans = 0
#     for pos in turb_coords:
#         if 50 <= pos <= 3950:
#             continue
#         else:
#             ans += max(abs(pos-50),abs(pos-3950))

#     return ans/4000

def proxi_constraint(turb_coords):
    """
        Function to penalize if proximity contraint is violated.
        turb_coords is a 2d numpy array with N (xi,yi) elements.
    """
    ans = 0
    for i in range(49):
        for j in range(i+1,50):
            norm = np.linalg.norm(turb_coords[i]-turb_coords[j])
            ans += norm
            
    return 1/ans


def obj_util(turb_coords, turb_rad, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, a, b):
    """
        Objective function to be minimized w.r.t. turb_coords.
        a,b are the hyperparameters to adjust the peri_val and proxi_val contributions
    """

    # peri_val = peri_constraint(turb_coords)

    turb_coords = np.array([[turb_coords[i],turb_coords[i+1]] for i in range(0,99,2)])
    AEP = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

    proxi_val = proxi_constraint(turb_coords)

    # return 1/AEP + a*peri_val + b*proxi_val
    return 1/AEP + b*proxi_val


def obj(turb_coords, kwargs):
    ans = []
    for turb_candidate in turb_coords:
        ans.append(obj_util(turb_candidate, **kwargs))

    return np.array(ans)


if __name__ == '__main__':
    # setting turbine radius
    turb_rad = 50.0
    # Loading the power curve
    power_curve   =  loadPowerCurve('Shell_Hackathon Dataset/power_curve.csv')
    # Loading wind data 
    wind_inst_freq =  binWindResourceData('Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)

    # hyperparameter options for optimizer c1: , c2: social parameter, w:inertia
    options = {'c1' : 0.3, 'c2' : 0.6 , 'w' : 0.8}

    bounds = tuple([50*np.ones(100),3950*np.ones(100)])

    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=50,dimensions=100,options=options,bounds=bounds)

    
    kwargs={'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freq': wind_inst_freq,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t, 'a': 0, 'b': 10000}    

    
    cost, pos = optimizer.optimize(obj, iters=1000, kwargs=kwargs)

    pos = np.array([[pos[i],pos[i+1]] for i in range(0,99,2)])
    checkConstraints(pos,100.0)
    print(getAEP(turb_rad, pos, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
    plt.scatter(pos[:,0],pos[:,1])
    plt.show()




