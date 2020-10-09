import matplotlib.pyplot as plt
import numpy as np
import EA.deap_spea as ea

# Number of Turbines to placed in the windfarm. 
N_TURB = 500

# Total Number of Population to be used for the Evolution
N_POP = 64

# Max number of generations to perform the Evolution for
MAX_GEN = 500

# Cross Probability adn Mutation Probabilities
CXPB = 0.7
MUTPB = 0.35
CXindpb = 0.8
MUTindpb = 0.4

# Resding Inputs and Saving Outputs 
IN_FILE_PATH = "EA/initData.csv"
OUT_FILE_DIR = "EA/outs/"
MODE = 'readcsv'

# loop_list = np.arange(0.1,0.6,0.05)
# max_fit_list = []
# for MUTindpb in loop_list:
fitness_history = ea.mainOptimize(N_TURB = N_TURB, N_POP = N_POP, MAX_GEN = MAX_GEN, 
        IN_FILE_PATH = IN_FILE_PATH, OUT_FILE_DIR = OUT_FILE_DIR, MODE = MODE, MultiProcess = False,
        CXPB = CXPB, MUTPB = MUTPB, CXindpb = CXindpb, MUTindpb = MUTindpb)
    # print(fitness_history)
    # max_fit_list.append(max(fitness_history))
# print(max_fit_list)
plt.plot(fitness_history)
plt.savefig("EA/plot100Gen.png", dpi = 300, bbox_inches = 'tight')
plt.show()
