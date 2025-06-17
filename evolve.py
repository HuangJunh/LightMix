# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import configparser
def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)

pop_size = int(__read_ini_file('SEARCH', 'pop_size'))
dataset = str(__read_ini_file('DATA', 'dataset'))
if dataset.__contains__('cifar'):
    from genotypes import search_space_cifar as search_space
else:
    from genotypes import search_space_imagenet as search_space

def reproduction(population, fitnessSet, params, _log):
    crossover_prob = params['crossover_prob']
    mutation_prob = params['mutation_prob']

    crossover = Crossover(population, fitnessSet, crossover_prob, _log)
    offspring = crossover.do_crossover_Structure()
    
    mutation = Mutation(offspring, mutation_prob, _log)
    offspring = mutation.do_mutation_Structure()

    return offspring


class Crossover(object):
    def __init__(self, individuals, fitnessSet, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log
        self.fitnessSet = fitnessSet

    def _choose_one_parent(self):
        count_ = pop_size
        idx1 = np.random.randint(0, count_)
        idx2 = np.random.randint(0, count_)
        while idx2 == idx1:
            idx2 = np.random.randint(0, count_)

        if self.fitnessSet[idx1] >= self.fitnessSet[idx2]:
            return idx1
        else:
            return idx2
    """
    binary tournament selection
    """
    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()

        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    def do_crossover_Structure(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        for _ in range(pop_size // 2):
            index1, index2 = self._choose_two_diff_parents()
            parent1, parent2 = copy.deepcopy(self.individuals[index1]), copy.deepcopy(self.individuals[index2])
            parent1_Structure, parent2_Structure = parent1, parent2
            p_ = np.random.random()

            if p_ < self.prob:
                offspring1_Structure = []
                offspring2_Structure = []
                num_genes = len(parent1)
                mask = np.random.random(num_genes)
                for i in range(num_genes):
                    _stat_param['offspring_new'] += 2
                    # multi-point crossover
                    if mask[i] <= 0.5:
                        offspring1_Structure.append(parent1_Structure[i])
                        offspring2_Structure.append(parent2_Structure[i])
                    else:
                        offspring1_Structure.append(parent2_Structure[i])
                        offspring2_Structure.append(parent1_Structure[i])

                parent1, parent2 = offspring1_Structure, offspring2_Structure
                self.log.info('Performing multi-point crossover for structure')
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)
            else:
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        self.log.info('CROSSOVER_Structure-%d offspring are generated, new:%d, others:%d' % (
        len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list



class Mutation(object):

    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log

    def do_mutation_Structure(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}

        offspring = []
        for indi in self.individuals:
            names = search_space['names']
            is_mutated = False
            for i, name in enumerate(names):
                p_ = np.random.random()
                if p_ < self.prob:
                    # [s, e, k, se, f]
                    mutatePoint = i
                    is_mutated = True
                    indi[mutatePoint][0] = float(np.random.choice(search_space[name][0]))
                    indi[mutatePoint][1] = int(np.random.choice(search_space[name][1]))
                    indi[mutatePoint][2] = int(np.random.choice(search_space[name][2]))
                    indi[mutatePoint][3] = float(np.random.choice(search_space[name][3]))
                    indi[mutatePoint][4] = float(np.random.choice(search_space[name][4]))

            if is_mutated:
                _stat_param['offspring_new'] += 1
            else:
                _stat_param['offspring_from_parent'] += 1
            offspring.append(indi)
        self.log.info('MUTATION_Structure-mutated individuals:%d, no_changes:%d' % (
        _stat_param['offspring_new'], _stat_param['offspring_from_parent']))

        return offspring

