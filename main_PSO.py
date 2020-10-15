from PSO.utils import *

if __name__ == '__main__':
    # no of turbines
    n_turbs = 50

    # swarm size
    n_part = 16


    # getting initial swarm
    oswarm = get_init(n_turbs, n_part=n_part, smart=True)

    # getting values for aep calculation
    # years = ['07']#,'08','09','13','14','15','17']
    years = ['08','09','13','14','15','17']
    kwargs = parse_data_PSO(n_turbs, years)

    # # pso global optimizer parameters
    # k = 1    # damping factor
    # w = 0.72984    # inertia
    # c1 = 0.5*(w + 1)**2   # cognitive
    # c2 = c1   # social

    # pso global optimizer parameters
    c1 = 0.4   # cognitive  0.35   0.1
    c2 = 3  # social     3.05   2.3
    w = 0.003  # inertia
    k = 4    #  1.3    1.8
    

    fig, ax = plt.subplots()
    for i in range(1):
        # creating the optimizer
        optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=oswarm)
        # optimizing
        cost, pos = optimizer.optimize(obj, iters=500, n_processes=4, verbose=True, damping=k, kwargs=kwargs)
        # oswarm = optimizer.swarm.position
        # oswarm[np.random.randint(n_part),:] = pos
        print('current AEP is ', -kwargs['ideal_AEP']*cost)
        print('ideal AEP is', kwargs['ideal_AEP'])
        plot_cost_history(optimizer.cost_history, ax=ax)


    # visualizing the optimized answer
    pos = pos.reshape((n_turbs,2))
    fig = plt.figure(figsize=(6,6))
    ax2 = plt.gca()
    ax2.set_aspect(1)
    ax2.scatter(pos[:,0],pos[:,1], s=900)
    plt.show()
