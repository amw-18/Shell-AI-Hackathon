import numpy as np
import pandas as pd
import random
import shapely
from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from shapely.ops import nearest_points 
from shapely import wkt

from Wind_Farm_Evaluator.Vec_modified import *

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
    turb_list = []
    count = 0
    while count < n_turbs:
        point = [get_point(),get_point()] # x,y
        if is_valid(point):
            turbine_pos[count,:] = point
            count += 1
            # turb_list.append(point)
    return turbine_pos

def get_arranged_location(data_file):
    ans = []
    locs = np.array(pd.read_csv(data_file))
    for loc in locs:
        ans.append(loc.reshape((50,2)))
    return np.array(ans)


def parse_data(n_turbs):
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

    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t}

    return kwargs

def evaluateAEP(individual):#, n_turbs, turb_rad, power_curve, wind_inst_freqs, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t):
    """
        Function to return values of both objectives as a tuple
    """
    n_turbs = len(individual)
    # rearranging the terms in individual to pass to getAEP
    turb_coords = np.array([[individual[i][0],individual[i][1]] for i in range(0,n_turbs)])
    data = parse_data(n_turbs)
    # calculating meanAEP for 
    mean_AEP = 0
    for wind_inst_freq in data['wind_inst_freqs']:
        mean_AEP += getAEP(data['turb_rad'], turb_coords, data['power_curve'], wind_inst_freq, data['n_wind_instances'], data['cos_dir'], data['sin_dir'], data['wind_sped_stacked'], data['C_t'])
    mean_AEP /= len(data['wind_inst_freqs'])
    
    ideal_AEP = 574.64  # 11.297 is the mean score for 1 turbine
    
    return mean_AEP/ideal_AEP, # First objective should be closest to 1 and second closest to zero


def checkBounds(Min, Max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                field = Polygon([(Min,Min), (Min,Max), (Max,Max), (Max,Min)])
                for i,point in enumerate(child):
                    pt = Point(point)
                    if field.contains(pt):
                        pass
                    else:
                        pt_new,_ = nearest_points(field,pt)
                        pt = Point(pt_new)
                        child[i] = np.array([pt.x, pt.y])
                    field = field.difference(pt.buffer(401))                   
            return offspring
        return wrapper
    return decorator



