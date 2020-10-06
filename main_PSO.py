from PSO.utils import *


if __name__ == '__main__':
    # no of turbines
    n_turbs = 50

    # swarm size
    n_part = 16

    # getting initial swarm
    oswarm = get_init(50, n_part=n_part, smart=True)
    # oswarm = np.zeros((n_part, 100))
    # for i in range(n_part):
    #     oswarm[i,:] = np.array(pd.read_csv(f'C:/Users/awals/Downloads/Shell AI Hackathon/Trials/opt_swarm_ans{i+20}.csv')).flatten()

    # oswarm[-2,:] = np.array(pd.read_csv('C:/Users/awals/Downloads/Shell AI Hackathon/Trials/opt_swarm_ans23.csv')).flatten()
    # oswarm[-1,:] = np.array(pd.read_csv('C:/Users/awals/Downloads/Shell AI Hackathon/Trials/opt_swarm_ans33.csv')).flatten()
    fig2, axs2 = plt.subplots(4, 4)
    for i in range(16):
        part = oswarm[i].reshape((50,2))
        axs2[i//4][i%4].scatter(part[:,0],part[:,1])

    plt.show()

    # getting values for aep calculation
    kwargs = parse_data_PSO(n_turbs)

    # pso global optimizer parameters
    c1 = 0.3   # cognitive  0.35   0.1
    c2 = 3   # social     3.05   2.3 
    w = 0.03  # inertia
    k = 10     #          1.3    1.8

    fig, ax = plt.subplots()
    for i in range(1):
        # creating the optimizer
        optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=oswarm)
        # optimizing
        cost, pos = optimizer.optimize(obj, iters=1000, n_processes=12, verbose=True, damping=k, kwargs=kwargs)
        # oswarm[np.random.randint(n_part),:] = pos
        # oswarm = optimizer.swarm.position
        print('current AEP is ', -kwargs['ideal_AEP']*cost)
        print('ideal AEP is', kwargs['ideal_AEP'])
        plot_cost_history(optimizer.cost_history, ax=ax)
        # w *= 0.6
        # c1 *= 0.75 

    fig2, axs2 = plt.subplots(4, 4)
    for i in range(16):
        part = oswarm[i].reshape((50,2))
        axs2[i//4][i%4].scatter(part[:,0],part[:,1])

    plt.show()
    pos = pos.reshape((50,2))

    fig, ax2 = plt.subplots()
    ax2.scatter(pos[:,0],pos[:,1])
    plt.show()
