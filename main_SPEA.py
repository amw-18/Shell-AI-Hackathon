import matplotlib.pyplot as plt
import numpy as np
import EA.deap_spea as ea

# Number of Turbines to placed in the windfarm. 
N_TURB = 50

# Total Number of Population to be used for the Evolution
N_POP = 64

# Max number of generations to perform the Evolution for
MAX_GEN = 10

# Cross Probability adn Mutation Probabilities
CXPB = 0.4
MUTPB = 0.1
CXindpb = 0.6
MUTindpb = 0.2

# Resding Inputs and Saving Outputs 
IN_FILE_PATH = "EA/initData.csv"
OUT_FILE_DIR = "EA/outs/"
MODE = 'readcsv'

CXPB_list = np.arange(0.1,0.9,0.1)
max_fit_list = []
for CXPB in CXPB_list:
    fitness_history = ea.mainOptimize(N_TURB = N_TURB, N_POP = N_POP, MAX_GEN = MAX_GEN, 
        IN_FILE_PATH = IN_FILE_PATH, OUT_FILE_DIR = OUT_FILE_DIR, MODE = MODE, MultiProcess = False,
        CXPB = CXPB, MUTPB = MUTPB, CXindpb = CXindpb, MUTindpb = MUTindpb)
    print(fitness_history)
    max_fit_list.append(max(fitness_history))
print(max_fit_list)
plt.plot(CXPB_list,max_fit_list)
plt.show()
