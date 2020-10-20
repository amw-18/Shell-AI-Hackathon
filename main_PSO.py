from PSO.utils import *

if __name__ == '__main__':
    # no of turbines
    n_turbs = 46

    # swarm size
    n_part = 40


    # getting initial swarm
    oswarm = get_init(n_turbs, n_part=n_part, smart=True)
    # oswarm[0,:] = np.array(pd.read_csv("Trials/trial2.csv")).flatten()
    # oswarm[1,:] = np.array(pd.read_csv("PSO/FINAL_BRUTE.csv")).flatten()
    # oswarm[2,:] = np.array(pd.read_csv("PSO/brute/50.csv")).flatten()
    # oswarm[3,:] = np.array(pd.read_csv("PSO/brute2/1_50.csv")).flatten()
    # oswarm[4,:] = np.array(pd.read_csv("PSO/brute2/2_50.csv")).flatten()
    # oswarm[5,:] = np.array(pd.read_csv("PSO/brute3/50.csv")).flatten()
    # oswarm[6,:] = np.array(pd.read_csv("PSO/brute4/50.csv")).flatten()
    # oswarm[7,:] = np.array(pd.read_csv("PSO/brute5/50.csv")).flatten()

    # getting values for aep calculation
    years = ['07','08','09','13','14','15','17']
    # years = ['17']
    kwargs = parse_data_PSO(n_turbs, years)

    # pso global optimizer parameters
    k = 1   # damping factor
    w = 0.7296    # inertia
    c1 = 0.5*(w + 1)**2   # cognitive
    c2 = c1   # social

    # # pso global optimizer parameters
    # c1 = 0.4   # cognitive  0.35   0.1
    # c2 = 3  # social     3.05   2.3
    # w = 0.004  # inertia
    # k = 4    #  1.3    1.8
    

    fig, ax = plt.subplots()
    for i in range(2):
        # creating the optimizer
        optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=oswarm)
        # optimizing
        cost, pos = optimizer.optimize(obj, iters=1000, n_processes=12, verbose=True, damping=k, kwargs=kwargs)
        print(-np.mean(obj(optimizer.swarm.position, kwargs)))
        oswarm = optimizer.swarm.position
        oswarm[np.random.randint(n_part),:] = pos
        print('current AEP is ', -kwargs['ideal_AEP']*cost)
        print('ideal AEP is', kwargs['ideal_AEP'])
        plot_cost_history(optimizer.cost_history, ax=ax)
        w = 0.004
        c1 = 0.4
        c2 = 3

    # visualizing the optimized answer
    # pos = pos.reshape((n_turbs,2))
    # fig = plt.figure(figsize=(6,6))
    # ax2 = plt.gca()
    # ax2.set_aspect(1)
    # ax2.scatter(pos[:,0],pos[:,1], s=900)
    # plt.show()



# if __name__ == '__main__':
#     # no of turbines
#     n_turbs = 50

#     n_opt_part = 64

#     # swarm size
#     n_part = 64

#     # getting values for aep calculation
#     # years = ['07']#,'08','09','13','14','15','17']
#     years = ['08','09','13','14','15','17']
#     kwargs = parse_data_PSO(n_turbs, years)

#     # pso global optimizer parameters
#     k = 1    # damping factor
#     w = 0.72984    # inertia
#     c1 = 0.5*(w + 1)**2   # cognitive
#     c2 = c1   # social

#     # # pso global optimizer parameters
#     # c1 = 0.4   # cognitive  0.35   0.1
#     # c2 = 3  # social     3.05   2.3
#     # w = 0.003  # inertia
#     # k = 4    #  1.3    1.8
    
#     opt_swarm = np.empty((n_opt_part, 100))
#     fig, ax = plt.subplots()
#     for i in range(n_opt_part):
#         # getting initial swarm
#         oswarm = get_init(n_turbs, n_part=n_part, smart=False)
#         # creating the optimizer
#         optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=oswarm)
#         # optimizing
#         cost, pos = optimizer.optimize(obj, iters=25, n_processes=4, verbose=False, damping=k, kwargs=kwargs)

#         opt_swarm[i,:] = pos
#         # oswarm = optimizer.swarm.position
#         # oswarm[np.random.randint(n_part),:] = pos
#         print(f'{i}th AEP is ', -kwargs['ideal_AEP']*cost)
#         # print('ideal AEP is', kwargs['ideal_AEP'])  # 562.84 for all but 07
#         plot_cost_history(optimizer.cost_history, ax=ax)

#     opt_swarm = pd.DataFrame(opt_swarm, index=None, columns=None)
#     opt_swarm.to_csv("PSO/oswarms/re_oswarm_0.csv", index=None, columns=None)

#     plt.show()



# current AEP is  421.41265071365495

# current AEP is  431.7379718736261

# current AEP is  441.72832466711196 ## 0.46

# current AEP is  451.8058181163781 ## 0.63

# current AEP is  461.9493505340189  ## 0.46

# current AEP is  471.8984809066559   ## 0.47

# current AEP is  481.8292400676834  ## 0.47

# current AEP is  491.7446518025497  ## 0.4

# current AEP is  501.62007486917264 ## 0.35

# current AEP is  511.2628496576083 ## 0.46

# current AEP is  521.0449146304829 ## 0.35

# current AEP is  530.7669859084266 ## 0.3

