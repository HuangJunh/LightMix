# !/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import configparser

def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)

dataset = str(__read_ini_file('DATA', 'dataset'))
if dataset.__contains__('cifar'):
    from genotypes import search_space_cifar as search_space
else:
    from genotypes import search_space_imagenet as search_space

def initialize_population(params):
    pop_size = params['pop_size']
    population = []
    names = search_space['names']
    # [s, e, k, se, f]
    # [[split ratio s], [expansion rate e], [kernel_combination], [SE ratio], [scale factors for f], [number of filters f], expand_ksize, project_ksize, stride]
    for _ in range(pop_size):
        structure = []
        p_len = np.random.random()
        for i, name in enumerate(names):
            if 0 in search_space[name][0]:
                p_ = np.random.random()
                if p_ > p_len:
                    s = 0.0
                else:
                    s_options = copy.deepcopy(search_space[name][0])
                    s_options.remove(0)
                    s = np.random.choice(s_options)
            else:
                s = np.random.choice(search_space[name][0])

            # s = np.random.choice(search_space[name][0])
            e = np.random.choice(search_space[name][1])
            k = np.random.choice(search_space[name][2])
            se = np.random.choice(search_space[name][3])
            f = np.random.choice(search_space[name][4])
            structure.append([s, e, k, se, f])

        population.append(structure)
    return population

def test_population():
    params = {}
    params['pop_size'] = 10

    pop = initialize_population(params)
    print(pop)
    print(pop[0])


if __name__ == '__main__':
    test_population()

