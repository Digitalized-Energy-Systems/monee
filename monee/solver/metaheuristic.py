import numpy as np
import copy

INDEX_SIGMA = -1
INDEX_FITNESS = -2

def is_productive():
    pass

class Evaluator:
    pass

class EASolver:
    def __init__(
        self,
        fitness_evaluator: Evaluator,
        population_size=16,
        generation_size=8,
        parent_number=2,
        iteration_number=100,
    ) -> None:
        assert parent_number <= population_size

        self._fitness_evaluator = fitness_evaluator
        self._population_size = population_size
        self._generation_size = generation_size
        self._parent_number = parent_number
        self._iteration_number = iteration_number

    def _init_population(self, solution_length):
        sigma_array = np.array(
            [
                [
                    np.exp(0.22 * np.random.normal(0, 1))
                    for _ in range(self._population_size)
                ]
            ]
        )
        fitness_array = np.array(
            [[-float("inf") for _ in range(self._population_size)]]
        )

        return np.concatenate(
            (
                np.random.rand(self._population_size, solution_length),
                fitness_array.transpose(),
                sigma_array.transpose(),
            ),
            axis=1,
        )

    def _select_parents(self, population):
        population_copy = np.array(population)
        np.random.shuffle(population_copy)
        return population_copy[0 : self._parent_number]

    def _recombine(self, solutions):
        return solutions.sum(axis=0) / len(solutions)

    def _mutate(self, solution):
        solution[INDEX_SIGMA] = solution[INDEX_SIGMA] * np.exp(
            0.22 * np.random.normal(0, 1)
        )

        raw_mutation = np.random.rand(len(solution))
        raw_mutation[INDEX_SIGMA] = 0
        raw_mutation[INDEX_FITNESS] = 0
        mutation = (raw_mutation - 0.5) * 0.3 * solution[INDEX_SIGMA]
        mutated_solution = np.clip(solution + mutation, 0, 1)
        mutated_solution[INDEX_SIGMA] = solution[INDEX_SIGMA]
        mutated_solution[INDEX_FITNESS] = solution[INDEX_FITNESS]
        return mutated_solution

    def _evaluate(self, solution, me_network, all_regulatable_nodes, step):
        solution[INDEX_FITNESS] = self._fitness_evaluator.evaluate(
            solution[0:-2], me_network, all_regulatable_nodes, step
        )

    def _select(self, population):
        start = len(population) - self._population_size
        return population[population[:, INDEX_FITNESS].argsort()][start:]

    def solve(self, me_network, step: int, without_load=False):

        all_regulatable_nodes = [
            node
            for node in me_network.nodes
            if is_productive(node, without_load=without_load)
        ]

        population = self._init_population(len(all_regulatable_nodes))

        best = None
        fitness_history = []
        for _ in range(self._iteration_number):
            generation = None
            for _ in range(self._generation_size):
                parents = self._select_parents(population)
                new_solution = self._recombine(parents)
                mutated_solution = self._mutate(new_solution)
                self._evaluate(
                    mutated_solution, me_network, all_regulatable_nodes, step
                )

                if generation is None:
                    generation = mutated_solution
                else:
                    generation = np.vstack((generation, mutated_solution))

            population = np.concatenate((population, generation))
            population = self._select(population)

            if best is None or best[INDEX_FITNESS] < population[-1][INDEX_FITNESS]:
                best = population[-1]
                fitness_history.append(best[INDEX_FITNESS])

        return best[0:-2], best[-2], fitness_history
