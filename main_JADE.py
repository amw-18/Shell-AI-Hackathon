from EA.jade import *

def update_ind(individual, pop, mu1, mu2, n_jrand):
    """
        updates the individual
        pop: list of individuals in population sorted according to fitness values
        returns an updated copy of the individual and parameter values that resulted
        in successful update
    """
    global toolbox
    N = len(individual)

    # selecting the best
    s_best = pop[-1]

    # selecting a unique random individual
    while 1:
        s1 = random.choice(pop[:-1])
        if s1 is not individual:
            break

    # creating a distorted individual out of the selected one
    s2 = creator.Individual(random.sample(list(s1), N))

    # getting -1 or 1 randomly
    sgn = np.sign(random.uniform(-1, 1))

    # getting the parameters to use
    F1, F2 = random.normalvariate(mu1, 1), random.normalvariate(mu2, 1)

    # computing the velocity of the particle
    D = sgn*(F1*s1 - F2*s2)
    v = s_best + D

    # getting the number of turbs to cross
    ni_jrand = int(random.normalvariate(n_jrand, 5))
    ni_jrand = max(1,min(N,ni_jrand))

    # crossing the turbs
    updated_individual = creator.Individual([*random.sample(list(v), ni_jrand), *random.sample(list(individual), N-ni_jrand)])
    
    # repairing the updated individual inplace if contraints violated
    repair(updated_individual)

    # evaluating and providing fitness value to the updated individual
    toolbox.evaluate(updated_individual)

    # selecting the better individual
    if individual.fitness.dominates(updated_individual.fitness):
        return individual, None, None, None
    else:
        return updated_individual, F1, F2, ni_jrand


if __name__ == "__main__":
    n_turbs = 50
    pop_size = 16
    is_smart = True
    n_gens = 1000

    means = []
    bests = []

    # jd.mainOptimize(n_turbs, pop_size, is_smart)

    years = ['08','09','13','14','15','17']#,'07']
    kwargs = parse_data(n_turbs, years)
    #--------------------
    # Create Base Classes
    #--------------------
    creator.create("Fitness", base.Fitness, weights=(1.0,)) 
    creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, n_turbs, is_smart)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)
    toolbox.register("evaluate", evaluateAEP, data=kwargs)
    toolbox.register("map", map)
    toolbox.register("update", update_ind)

    mu1, mu2, n_jrand = 1, 1, 25
    pop = toolbox.population()
    # pop = initPopulation(list, creator.Individual, r"C:\Users\awals\Downloads\Shell AI Hackathon\PSO\opt_swarm_smart.csv")
    list(toolbox.map(toolbox.evaluate, pop))
    # sorting the population so that best individual is last
    pop.sort(key=lambda x: x.fitness.values[0])

    best_fitness = pop[-1].fitness.values[0]
    print(f"---Generation {0}---")
    print("Population best...", best_fitness)

    old_best = best_fitness
    for gen in range(n_gens):
        print(f"|--- Generation {gen+1} ---|")

        # mutation and cross operation
        ans = list(toolbox.map(update_ind, pop, [pop]*pop_size,
             [mu1]*pop_size, [mu2]*pop_size, [n_jrand]*pop_size))

        # updating population
        pop = [ind[0] for ind in ans]

        # getting successful values of parameters
        S_f1 = [ind[1] for ind in ans if ind[1] is not None]
        S_f2 = [ind[2] for ind in ans if ind[2] is not None]
        S_n_jrand = [ind[3] for ind in ans if ind[3] is not None]
        
        # updating parameter values if new successful found
        if S_f1:
            mu1 = np.mean(S_f1) 
        if S_f2:
            mu2 = np.mean(S_f2)
        if S_n_jrand:
            n_jrand = np.mean(n_jrand)

        # sorting the population so that best individual is last
        pop.sort(key=lambda x: x.fitness.values[0])
        best_fitness = pop[-1].fitness.values[0]
        bests.append(best_fitness)

        if best_fitness > old_best:
            print("New best...", best_fitness)
            old_best = best_fitness
        
        fitnesses = [ind.fitness.values[0] for ind in pop]
        mean_fitness = np.mean(fitnesses)
        means.append(mean_fitness)
        print("Population mean...", mean_fitness)
        

    plt.plot(range(n_gens), bests, label="best")
    plt.plot(range(n_gens), means, label="population mean")
    plt.xlabel("generation")
    plt.ylabel("fitness value")
    plt.legend()

    print("\nEvolution best fitness", old_best)
    print(pop[-1])
    plt.savefig("ADE_fig2.png")