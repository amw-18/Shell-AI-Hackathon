
import pyswarms as ps 
from pyswarms.utils.functions import single_obj as fx
import numpy as np 
from multiprocessing import freeze_support

if __name__ == '__main__':
    # freeze_support()
    # Set-up hyperparameters
    options = {'c1': 1.49, 'c2': 1.49, 'w': 0}

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.global_best.GlobalBestPSO(n_particles=64, dimensions=2, ftol=1e-16,                                               ftol_iter=50, options=options, bounds=([-512]*2,[512]*2))

    # Perform optimization
    stats = optimizer.optimize(fx.eggholder, iters=1000, n_processes=8)



