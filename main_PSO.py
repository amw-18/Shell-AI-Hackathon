from PSO.utils import *


if __name__ == '__main__':
    # no of turbines
    n_turbs = 50

    # swarm size
    n_part = 64

    # getting initial swarm
    oswarm = get_init(50, 64)

    # getting values for aep calculation
    kwargs = parse_data_PSO(n_turbs)

    # pso global optimizer parameters
    c1 = 0.2   # cognitive
    c2 = 3   # social
    w = 0.002  # inertia

    fig, ax = plt.subplots()
    for i in range(1):
        # creating the optimizer
        optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, v_clamp=True, init_vals=oswarm)
        # optimizing
        cost, pos = optimizer.optimize(obj, iters=300, n_processes=12, verbose=True, damping=3, kwargs=kwargs)
        print('current AEP is ', -kwargs['ideal_AEP']*cost)
        plot_cost_history(optimizer.cost_history, ax=ax)
        # print(pos)
    plt.show()
