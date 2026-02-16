import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class IslandGeneticAlgorithm:

    def __init__(
        self,
        param_lists,
        fitness_function,
        fitness_context=None,
        islands=5,
        pop_size=20,
        generations=5,
        migration_interval=3,
        elite_size=3,
        mutation_prob=0.7,
        tournament_k=3,
        n_jobs=None,
        verbose=True
    ):

        self.param_lists = param_lists
        self.fitness_function = fitness_function
        self.context = fitness_context

        self.islands = islands
        self.pop_size = pop_size
        self.generations = generations
        self.migration_interval = migration_interval

        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.tournament_k = tournament_k
        self.n_jobs = n_jobs

        self.verbose = verbose

        self.island_pops = [
            [self.generate_individual() for _ in range(pop_size)]
            for _ in range(islands)
        ]

        self.fitness_history = []
        self.best_params_per_gen = []

    # Generate individual
    def generate_individual(self):

        return [
            random.choice(param)
            for param in self.param_lists
        ]

    # Mutation
    def mutate(self, individual):

        ind = individual.copy()

        idx = random.randint(0, len(ind) - 1)

        ind[idx] = random.choice(self.param_lists[idx])

        return ind

    # Crossover
    def crossover(self, parents):

        return [
            random.choice(gene)
            for gene in zip(*parents)
        ]

    # Tournament selection
    def tournament_selection(self, population, fitness_scores):

        tournament = random.sample(
            list(zip(population, fitness_scores)),
            self.tournament_k
        )

        return max(tournament, key=lambda x: x[1])[0]

    # Evaluate population
    def evaluate_population(self, population):

        if self.context is None:

            results = [
                self.fitness_function(ind)
                for ind in population
            ]

        else:

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:

                results = list(
                    executor.map(
                        lambda ind: self.fitness_function(ind, self.context),
                        population
                    )
                )

        # SAFELY handle scalar OR tuple fitness return
        fitness_scores = []

        for r in results:

            if isinstance(r, (list, tuple, np.ndarray)):
                fitness_scores.append(r[0])
            else:
                fitness_scores.append(r)

        return fitness_scores


    # Migration
    def migrate(self):

        migrants = [
            random.choice(pop)
            for pop in self.island_pops
        ]

        for i in range(len(migrants)):

            target = (i + 1) % len(self.island_pops)

            self.island_pops[target][-1] = migrants[i]

    # Main run
    def run(self):

        best_fitness_global = -np.inf
        best_params_global = None

        for gen in range(self.generations):

            for island_idx in range(self.islands):

                population = self.island_pops[island_idx]

                fitness_scores = \
                    self.evaluate_population(population)

                best_idx = np.argmax(fitness_scores)

                best_fitness = fitness_scores[best_idx]
                best_params = population[best_idx]

                if best_fitness > best_fitness_global:

                    best_fitness_global = best_fitness
                    best_params_global = best_params

                if self.verbose:

                    print(  
                        f"Gen {gen+1:03d} | "
                        f"Island {island_idx+1:02d} | "
                        f"Fitness {best_fitness:.6f} | "
                        f"Params {best_params}"
                    )

                self.fitness_history.append(best_fitness)
                self.best_params_per_gen.append(best_params)

                sorted_pop = [
                    x for _, x in sorted(
                        zip(fitness_scores, population),
                        reverse=True
                    )
                ]

                next_gen = sorted_pop[:self.elite_size]

                while len(next_gen) < self.pop_size:

                    parents = [
                        self.tournament_selection(
                            population,
                            fitness_scores
                        )
                        for _ in range(3)
                    ]

                    child = self.crossover(parents)

                    if random.random() < self.mutation_prob:

                        child = self.mutate(child)

                    next_gen.append(child)

                self.island_pops[island_idx] = next_gen

            if (gen + 1) % self.migration_interval == 0:

                self.migrate()

        return best_params_global, best_fitness_global
