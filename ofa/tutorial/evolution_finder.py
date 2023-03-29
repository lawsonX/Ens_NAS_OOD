import copy
import random
from tqdm import tqdm
import numpy as np
from ofa.utils.pytorch_utils import count_parameters
from .imagenet_eval_helper import evaluate_ofa_subnet

__all__ = ["EvolutionFinder"]


class EvolutionFinder:
    valid_constraint_range = {
        "flops": [150, 600],
        "note10": [15, 60],
        "param": [1,100],
    }

    def __init__(
        self,
        constraint_type,
        efficiency_constraint,
        # efficiency_predictor,
        accuracy_predictor,
        ofa_network,
        # data_loader, # add 
        **kwargs
    ):  
        # self.data_loader = data_loader # add
        self.ofa_network = ofa_network
        self.constraint_type = constraint_type
        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_constraint = efficiency_constraint
        if not (
            efficiency_constraint <= self.valid_constraint_range[constraint_type][1]
            and efficiency_constraint >= self.valid_constraint_range[constraint_type][0]
        ):
            self.invite_reset_constraint()

        # self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    def invite_reset_constraint_type(self):
        print(
            "Invalid constraint type! Please input one of:",
            list(self.valid_constraint_range.keys()),
        )
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print(
                "Invalid constraint type! Please input one of:",
                list(self.valid_constraint_range.keys()),
            )
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        print(
            "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
            % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1],
            )
        )

        new_cons = input()
        while (
            (not new_cons.isdigit())
            or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1])
            or (int(new_cons) < self.valid_constraint_range[self.constraint_type][0])
        ):
            print(
                "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
                % (
                    self.valid_constraint_range[self.constraint_type][0],
                    self.valid_constraint_range[self.constraint_type][1],
                )
            )
            new_cons = input()
        new_cons = int(new_cons)
        self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint
    
    def compute_ens_efficiency(self, sample):
        """
        sum 越大 模型越小
        """
        sum = sum_of_lists(sample)
        return 25-sum

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.ofa_network.generate_random_config_list(pruning_rate_list=[round(x, 2) for x in np.arange(0, 0.6, 0.1)])
            subnet = self.ofa_network.set_network_from_config(sample)
            efficiency = self.compute_ens_efficiency(sample)
            acc = self.accuracy_validator(subnet)
            print('subnet size:' , efficiency)
            if efficiency <= constraint:
                return sample, efficiency, acc

    def mutate_sample(self, sample):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample)
            for i in range(len(new_sample)):
                for j in range(len(new_sample[i])):
                    if isinstance(new_sample[i][j], list):
                        for k in range(len(new_sample[i][j])):
                            new_sample[i][j][k] += random.choice(
                                [round(x, 2) for x in np.arange(new_sample[i][j][k]-0.1, new_sample[i][j][k]+0.1, 0.05)]
                                )
                    else:
                        new_sample[i][j] += random.choice(
                            [round(x, 2) for x in np.arange(new_sample[i][j][k]-0.1, new_sample[i][j][k]+0.1, 0.05)]
                            )
            
            subnet = self.ofa_network.set_network_from_config(new_sample)
            efficiency = self.compute_ens_efficiency(new_sample)
            acc = self.accuracy_validator(subnet)
            print('subnet size:' , efficiency)
            if efficiency <= constraint:
                return new_sample, efficiency, acc

    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample1)
            for i in range(len(new_sample)):
                for j in range(len(new_sample[i])):
                    if isinstance(new_sample[i][j], list):
                        for k in range(len(new_sample[i][j])):
                            new_sample[i][j][k] = random.choice([sample1[i][j][k], sample2[i][j][k]])
                    else:
                        new_sample[i][j] = random.choice([sample1[i][j], sample2[i][j]])

            subnet = self.ofa_network.set_network_from_config(new_sample)
            efficiency = self.compute_ens_efficiency(new_sample)
            acc = self.accuracy_validator(subnet)
            print('subnet size:' , efficiency)
            if efficiency <= constraint:
                return new_sample, efficiency, acc

    def accuracy_validator(self,subnet):
        # test the subnet on the test dataset (cifar10 val)
        acc = evaluate_ofa_subnet(
            subnet,
            self.data_loader,
            device='cuda:0')
        return acc

    def run_evolution_search(self, verbose=False,):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []  # (validation, sample, efficiency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None
        accs = []
        if verbose:
            print("Generate random population...")
        for _ in range(population_size):
            sample, efficiency, acc = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)
            accs.append(acc)

        for i in range(population_size):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="Searching with %s constraint (%s)"
            % (self.constraint_type, self.efficiency_constraint),
        ):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print("Iter: {} Acc: {}".format(iter - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []
            accs = []

            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample, efficiency, acc = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)
                accs.append(acc)

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample, efficiency, acc = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)
                accs.append(acc)

            accs = self.accuracy_validator(child_pool) # change
            for i in range(population_size):
                population.append((accs[i], child_pool[i], efficiency_pool[i]))

        return best_valids, best_info

def sum_of_lists(lists):
    total_sum = 0
    for sublist in lists:
        for elem in sublist:
            if isinstance(elem, list):
                total_sum += sum(elem)
            else:
                total_sum += elem
    return total_sum

