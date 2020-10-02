from PSO.pso_modified_wind import *
import numpy as np
import pandas as pd
from pyswarms.utils.plotters import plot_cost_history

if __name__ == "__main__":
    oswarm = np.array(pd.read_csv('PSO/oswarm0.csv'))
    n_turbs = 50

    # setting turbine radius
    turb_rad = 50.0

    # Loading the power curve
    power_curve   =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')

    # Loading wind data 
    years = ['07']
    wind_inst_freqs = []
    for y in years:
        wind_inst_freqs.append(binWindResourceData(f'./Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv'))


    # preprocessing the wind data to avoid repeated calculations
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_fs': wind_inst_freqs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t} 


    # getting ideal_AEP for normalization purpose
    ref_loc = get_random_arrangement(1)
    ideal_AEP = 0
    for wind_inst_freq in wind_inst_freqs:
        ideal_AEP += getAEP(turb_rad, ref_loc, power_curve, wind_inst_freq, *preProcessing(power_curve, 1))

    ideal_AEP /= len(wind_inst_freqs)
    ideal_AEP *= n_turbs

    # defining parameters for optimization
    a = 100  # weight for the proximity penalty -- critical only if random initialization done
    c2 = 1.1  # social
    w = 0.009  # inertia
    c1 = 0.001 # cognitive

    kwargs['n_turbs'] = n_turbs
    kwargs['a'] = a
    kwargs['ideal_AEP'] = ideal_AEP

    optimizer = my_optim(n_turbs, a, c1, c2, w, oswarm)
    cost, pos = optimizer.optimize(obj, iters=200, kwargs=kwargs, n_processes=12)
    AEP = -kwargs['ideal_AEP']*obj_util(pos, **kwargs)

    print('opt_swarm aep', AEP)
    plot_cost_history(optimizer.cost_history)
    plt.show()

    arrgmnt = pos.reshape((50, 2))
    plt.scatter(arrgmnt[:,0],arrgmnt[:,1])
    plt.show()



# if __name__ == "__main__":
#     n_turbs = 50

#     # setting turbine radius
#     turb_rad = 50.0

#     # Loading the power curve
#     power_curve   =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')

#     # Loading wind data 
#     years = ['07']
#     wind_inst_freqs = []
#     for y in years:
#         wind_inst_freqs.append(binWindResourceData(f'./Shell_Hackathon Dataset/Wind Data/wind_data_20{y}.csv'))


#     # preprocessing the wind data to avoid repeated calculations
#     n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve,n_turbs)

#     kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_fs': wind_inst_freqs,
#             'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
#             'C_t': C_t} 


#     # getting ideal_AEP for normalization purpose
#     ref_loc = get_random_arrangement(1)
#     ideal_AEP = 0
#     for wind_inst_freq in wind_inst_freqs:
#         ideal_AEP += getAEP(turb_rad, ref_loc, power_curve, wind_inst_freq, *preProcessing(power_curve, 1))

#     ideal_AEP /= len(wind_inst_freqs)
#     ideal_AEP *= n_turbs

#     # defining parameters for optimization
#     a = 100  # weight for the proximity penalty -- critical only if random initialization done
#     c2 = 1  # social
#     w = 0  # inertia
#     c1 = 0 # cognitive

#     kwargs['n_turbs'] = n_turbs
#     kwargs['a'] = a
#     kwargs['ideal_AEP'] = ideal_AEP

#     opt_swarm = []
#     best_score = 0
#     while len(opt_swarm) < 64:
#         optimizer = my_optim(n_turbs, a, c1, c2, w, get_init(n_turbs, 64))

#         cost,pos = optimizer.optimize(obj, iters=200, kwargs=kwargs, verbose=False, n_processes=12)

#         AEP = -kwargs['ideal_AEP']*obj_util(pos, **kwargs)
#         best_score = max(best_score, AEP)
#         if AEP > 517.3:
#             opt_swarm.append(pos)
#             print('*', end='')
#     print()

#     opt_swarm = np.array(opt_swarm)
#     optimizer = my_optim(n_turbs, a, c1, 1.1, 0.01, opt_swarm)
#     cost, pos = optimizer.optimize(obj, iters=200, kwargs=kwargs, n_processes=12)
#     AEP = -kwargs['ideal_AEP']*obj_util(pos, **kwargs)

#     print('random_swarm aep', best_score)
#     print('opt_swarm aep', AEP)

#     oswarm = pd.DataFrame(opt_swarm)
#     oswarm.to_csv("C:/Users/awals/Downloads/Shell AI Hackathon/PSO/oswarm0.csv",index=False)

#     ans = np.array(pos).reshape((50, 2))
#     turbines = pd.DataFrame(ans,columns=['x','y'])
#     turbines.to_csv("C:/Users/awals/Downloads/Shell AI Hackathon/Trials/opt_swarm_ans0.csv",index=False)