from PSO.utils import *


if __name__ == '__main__':
    # no of turbines
    n_turbs = 50

    # swarm size
    n_part = 64

    # getting values for aep calculation
    kwargs = parse_data_PSO(n_turbs)

    # pso global optimizer parameters
    c1 = 0   # cognitive
    c2 = 2   # social
    w = 0.002  # inertia

    # creating the optimizer
    optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, v_clamp=True)

    # optimizing
    cost, pos = optimizer.optimize(obj, iters=200, n_processes=8, verbose=True, kwargs=kwargs)
    
    plot_cost_history(optimizer.cost_history)
    plt.show()
