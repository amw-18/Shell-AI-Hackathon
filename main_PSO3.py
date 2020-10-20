from PSO.utils import *

if __name__ == '__main__':
    # swarm size
    n_part = 40

    determined = np.array(pd.read_csv('C:/Users/awals/Downloads/Shell AI Hackathon/PSO/brute7/49.csv'))

    for n_turbs in range(50, 51):
        # getting initial swarm
        oswarm = get_particles(n_turbs, n_part, determined, '1')
        # r_particle = oswarm[0].reshape((n_turbs, 2))
        # plt.scatter(r_particle[:,0], r_particle[:, 1])

        # getting values for aep calculation
        # years = ['07','08','09','17']
        years = ['07','08','09','13','14','15','17']
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
        

        # fig, ax = plt.subplots()
        for i in range(2):
            # creating the optimizer
            optimizer = get_optimizer(n_part, n_turbs, c1, c2, w, init_vals=oswarm)
            # optimizing
            cost, pos = optimizer.optimize(obj, iters=1000, n_processes=8, verbose=True, damping=k, kwargs=kwargs)
            # print(-np.mean(obj(optimizer.swarm.position, kwargs)))
            # oswarm = optimizer.swarm.position
            oswarm[np.random.randint(n_part),:] = pos
            print('current AEP is ', -kwargs['ideal_AEP']*cost)
            print('ideal AEP is', kwargs['ideal_AEP'])
            # plot_cost_history(optimizer.cost_history, ax=ax)
            w = 0.004
            c1 = 0.4
            c2 = 3

        determined = pos.reshape((n_turbs, 2))
        # plt.scatter(determined[:,0], determined[:,1])
        # plt.show()
        # df = pd.DataFrame(determined,columns=['x','y'])
        # df.to_csv(f"PSO/brute7/{n_turbs}.csv", index=False)
    # visualizing the optimized answer
    # pos = pos.reshape((n_turbs,2))
    # fig = plt.figure(figsize=(6,6))
    # ax2 = plt.gca()
    # ax2.set_aspect(1)
    # ax2.scatter(pos[:,0],pos[:,1], s=900)
    # plt.show()