import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points 
# from shapely import wkt

from Wind_Farm_Evaluator.Vec_modified import *


def get_random_arrangement(n_turbs, a=50, b=3950):
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


def get_smart_arrangement(n_turbs=50):
    """
        returns a solution with a fully filled border and 
        random arrangement inside(np.array with shape (n_turbs, 2))
    """
    n_bord = [8, 8, 8, 8]
    bord_vals = [np.linspace(50, 3950, n_bord[i]+2) for i in range(4)]
    left_bound = [np.array([50, val]) for val in bord_vals[0][1:-1]]
    top_bound = [np.array([val, 3950]) for val in bord_vals[1]]
    right_bound = [np.array([3950, val]) for val in bord_vals[2][1:-1]]
    bottom_bound = [np.array([val, 50]) for val in bord_vals[3]]

    ans = [*top_bound, *right_bound, *left_bound, *bottom_bound]

    remaining = n_turbs - len(ans)
    ans.extend(get_random_arrangement(remaining, a=450, b=3550))
    ans = np.array(ans)
    
    return ans


def parse_data(n_turbs, years, ignore=None):
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

    # creating a dictionary for extracted values
    kwargs = {'turb_rad': turb_rad, 'power_curve': power_curve, 'wind_inst_freqs': wind_inst_freqs, 'n_turbs': n_turbs,
            'n_wind_instances': n_wind_instances, 'cos_dir': cos_dir, 'sin_dir': sin_dir, 'wind_sped_stacked': wind_sped_stacked,
            'C_t': C_t, 'ideal_AEP': ideal_AEP, 'a': 100}

    return kwargs


def evaluateAEP(individual, data):
    """
        Function to return values of objective as a tuple
        of length 1
    """
    n_turbs = len(individual)
    
    # data = parse_data(n_turbs)
    
    AEP = getAEP(data['turb_rad'], individual, data['power_curve'], data["wind_inst_freqs"][0], data['n_wind_instances'], data['cos_dir'], data['sin_dir'], data['wind_sped_stacked'], data['C_t'])
    
    individual.fitness.values = (AEP/data["ideal_AEP"],)


def repair(individual):
    """
        repairs a given individual inplace
    """

    field = Polygon([(50, 50), (50, 3950), (3950, 3950), (3950, 50)])
    for i, point in enumerate(individual):
        pt = Point(point)
        if not field.contains(pt):
            pt,_ = nearest_points(field, pt)
            individual[i,:] = [pt.x, pt.y]
        
        field = field.difference(pt.buffer(400))


def get_arranged_location(data_file):
    ans = []
    locs = np.array(pd.read_csv(data_file))
    for loc in locs:
        ans.append(loc.reshape((50,2)))
    return np.array(ans)


def initPopulation(pcls,icls,data_file):
    ind_list = get_arranged_location(data_file)
    pop = pcls(icls(i) for i in ind_list)
    return pop


def is_valid(individual):
    """
        checks for validity of individual
    """

    field = Polygon([(50, 50), (50, 3950), (3950, 3950), (3950, 50)])
    for i, point in enumerate(individual):
        pt = Point(point)
        if not field.contains(pt):
            return False
        
        field = field.difference(pt.buffer(400)) 

    return True