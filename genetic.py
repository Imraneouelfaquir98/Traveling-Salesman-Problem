import numpy as np
from ypstruct import structure

class Genetic:
    def __init__(
        self, 
        distances,
        cities,
        start, 
        max_iter = 100, 
        nbr_pop = 50,
        beta = 1,
        pc = 1,
        gamma = 0.1,
        mu = 0.01,
        sigma = 0.1
    ):
        self.distances = distances
        self.cities    = cities
        self.start     = start
        self.max_iter  = max_iter
        self.nbr_pop   = nbr_pop
        self.beta      = beta
        self.pc        = pc
        self.gamma     = gamma
        self.mu        = mu
        self.sigma     = sigma
        self.nc        = int(np.round(pc*nbr_pop/2)*2)
        
        # Best Solution Ever Found
        self.best_sol = structure()
        self.best_sol.road = None
        self.best_sol.cost = np.inf

        # Initialize Population
        self.pop = self.best_sol.repeat(self.nbr_pop)
        for i in range(self.nbr_pop):
            self.pop[i].road = self.random_raod([i for i in range(len(distances)) if i != cities.index(self.start)])
            self.pop[i].cost = self.calculate_distance(self.pop[i].road)
            if self.pop[i].cost < self.best_sol.cost:
                self.best_sol = self.pop[i].deepcopy()

        # Best Cost of Iterations
        self.bestcost = np.empty(self.max_iter)
        
    def fit(self):
        for it in range(self.max_iter):
            costs = np.array([x.cost for x in self.pop])
            avg_cost = np.mean(costs)
            if avg_cost != 0:
                costs = costs/avg_cost
            probs = np.exp(-self.beta*costs)

            popc = []
            for _ in range(self.nc//2):
                # Perform Roulette Wheel Selection
                p1 = self.pop[self.roulette_wheel_selection(probs)]
                p2 = self.pop[self.roulette_wheel_selection(probs)]

                # Perform Crossover
                c1, c2 = self.crossover(p1, p2)

                # Evaluate First Offspring
                c1.cost = self.calculate_distance(c1.road)
                if c1.cost < self.best_sol.cost:
                    self.best_sol = c1.deepcopy()

                # Evaluate Second Offspring
                c2.cost = self.calculate_distance(c2.road)
                if c2.cost < self.best_sol.cost:
                    self.best_sol = c2.deepcopy()

                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)

            # Merge, Sort and Select
            self.pop += popc
            self.pop = sorted(self.pop, key=lambda x: x.cost)
            self.pop = self.pop[0:self.nbr_pop]

            # Store Best Cost
            self.bestcost[it] = self.best_sol.cost

            # Show best cost every 10 iterations
            if it % 10 == 0:
                print("Iteration {}: Best Cost = {}".format(it, self.bestcost[it]))
        return self.best_sol
    
    def crossover(self, p1, p2):
        c1 = p1.deepcopy()
        c2 = p1.deepcopy()
        cover_point = np.random.randint(1, p1.road.shape[0] - 1)

        tmp = c1.road[1:cover_point].copy()
        c1.road[1:c1.road.shape[0] - cover_point] = c1.road[cover_point : c1.road.shape[0] - 1]
        c1.road[  c1.road.shape[0] - cover_point  : c1.road.shape[0] - 1] = tmp

        tmp = c2.road[1:cover_point].copy()
        c2.road[1:c2.road.shape[0] - cover_point] = c2.road[cover_point : c2.road.shape[0] - 1]
        c2.road[  c2.road.shape[0] - cover_point  : c2.road.shape[0] - 1] = tmp
        return c1, c2
    
    def roulette_wheel_selection(self, p):
        c = np.cumsum(p)
        r = sum(p)*np.random.rand()
        ind = np.argwhere(r <= c)
        return ind[0][0]

    def random_raod(self, nodes):
        road = [self.cities.index(self.start)]
        while not not nodes:
            select = np.random.randint(0, len(nodes))
            road.append(nodes[select])
            nodes.remove(nodes[select])
        road.append(self.cities.index(self.start))
        return np.array(road)
    
    
    def calculate_distance(self, road):
        dist = 0
        for i in range(len(road) - 1):
            dist += self.distances[road[i]][road[i+1]]
        return dist