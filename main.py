# !/usr/bin/python
# -*- coding: utf-8 -*-
from utils import Utils, Log
from population import initialize_population
from evaluate import fitnessEvaluate, fitnessTest
from evolve import reproduction
from selection import Selection
import copy, os, time
import numpy as np
import configparser
import matplotlib.pyplot as plt
from threading import Thread
from Architecture import Architecture
import torch.backends.cudnn as cudnn
import torch


import argparse
parser = argparse.ArgumentParser("LightMix_training")
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--dataset', type=str, help='target dataset, e.g., cifar10, cifar100, imagenet')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size_search', type=int, default=128, help='batch size during search (for NASWOT)')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers when loading data')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--epoch_test', type=int, default=600, help='number of training epochs')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--drop_connect_rate', type=float, default=0.2, help='drop connect rate')
parser.add_argument('--script_name', type=str, default='', help='name of the script')
parser.add_argument('--test', default=False, action='store_true')
args = parser.parse_args()


def create_directory():
    dirs = ['./log', './populations', './scripts', './trained_models', './img_txt']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def fitness_evaluate(population, curr_gen):
    err_set = []
    n_parameters_set = []
    n_flops_set = []

    if args.dataset == 'cifar10':
        import load_dataset.data_loader as data_loader
        args.data = '../datasets/CIFAR10_data'
        args.num_workers = 4
    elif args.dataset == 'cifar10':
        import load_dataset.data_loader_cifar100 as data_loader
        args.data = '../datasets/CIFAR100_data'
        args.num_workers = 4
    else:
        import load_dataset.data_loader_imagenet as data_loader
        args.data = '../datasets/imagenet/ILSVRC2012'
        args.num_workers = 32
    trainloader, validate_loader = data_loader.get_train_valid_loader(args.data,
                                                                       batch_size=args.batch_size_search, augment=True,
                                                                       subset_size=1, valid_size=0.1,
                                                                       shuffle=True,
                                                                       random_seed=1000, show_sample=False,
                                                                       num_workers=args.num_workers, pin_memory=True)

    for i, indi in enumerate(population):
        architecture = Architecture(indi, args)
        err, n_parameters, n_flops = fitnessEvaluate(architecture, trainloader, args)
        # filenames.append(filename)
        err_set.append(err)
        n_parameters_set.append(n_parameters)
        n_flops_set.append(n_flops)
    fitnessSet = [calc_fitness(err_set[i]) for i in range(len(err_set))]
    return err_set, n_parameters_set, n_flops_set, fitnessSet

def evolve(population, fitnessSet, params):
    offspring = reproduction(population, fitnessSet, params, Log)
    return copy.deepcopy(offspring)

def environment_selection(parentAndPerformance, offspringAndPerformance, params):
    parents, errs_parent, NumParams_parent, flops_parent, fitnessSet_parent = parentAndPerformance
    offspring, errs_offspring, NumParames_offspring, flops_offspring, fitnessSet_offspring = offspringAndPerformance

    err_list = []
    numParams_list = []
    flops_list = []
    fitness_list = []
    indi_list = []
    for i,indi in enumerate(parents):
        indi_list.append(indi)
        err_list.append(errs_parent[i])
        numParams_list.append(NumParams_parent[i])
        flops_list.append(flops_parent[i])
        fitness_list.append(fitnessSet_parent[i])
    for j,indi in enumerate(offspring):
        indi_list.append(indi)
        err_list.append(errs_offspring[j])
        numParams_list.append(NumParames_offspring[j])
        flops_list.append(flops_offspring[j])
        fitness_list.append(fitnessSet_offspring[j])

    # find the largest one's index
    max_index = np.argmax(fitness_list)
    selection = Selection()
    selected_index_list = selection.binary_tournament_selection(fitness_list, k=params['pop_size'])
    if max_index not in selected_index_list:
        first_selectd_v_list = [fitness_list[i] for i in selected_index_list]
        min_idx = np.argmin(first_selectd_v_list)
        selected_index_list[min_idx] = max_index

    next_individuals = [indi_list[i] for i in selected_index_list]
    next_errs = [err_list[i] for i in selected_index_list]
    next_NumParams = [numParams_list[i] for i in selected_index_list]
    next_flops = [flops_list[i] for i in selected_index_list]
    next_fitnessSet = [fitness_list[i] for i in selected_index_list]

    return next_individuals, next_errs, next_NumParams, next_flops, next_fitnessSet


def calc_fitness(err, numParams=None, flops=None):
    return 1-err

def update_best_individual(population, err_set, num_parameters, flops, gbest):
    fitnessSet = [calc_fitness(err_set[i]) for i in range(len(population))]
    if not gbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_errSet = copy.deepcopy(err_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness = getGbest([pbest_individuals, pbest_errSet, pbest_params, pbest_flops])
    else:
        gbest_individual, gbest_err, gbest_params, gbest_flops = gbest
        gbest_fitness = calc_fitness(gbest_err)
        for i,fitness in enumerate(fitnessSet):
            if fitness >= gbest_fitness:
                gbest_fitness = copy.deepcopy(fitness)
                gbest_individual = copy.deepcopy(population[i])
                gbest_err = copy.deepcopy(err_set[i])
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])
    return [gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness]

def getGbest(pbest):
    pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest
    gbest_err = 1e8
    gbest_params = 10e6
    gbest_flops = 10e9
    gbest = None

    gbest_fitness = calc_fitness(gbest_err)
    pbest_fitnessSet = [calc_fitness(pbest_errSet[i]) for i in range(len(pbest_individuals))]

    for i,indi in enumerate(pbest_individuals):
        if pbest_fitnessSet[i] >= gbest_fitness:
            gbest = copy.deepcopy(indi)
            gbest_err = copy.deepcopy(pbest_errSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
            gbest_fitness = copy.deepcopy(pbest_fitnessSet[i])
    return gbest, gbest_err, gbest_params, gbest_flops, gbest_fitness


def split_population(population, objVals):
    population = copy.deepcopy(population)
    objVals = copy.deepcopy(objVals)
    sub_populations = [[] for _ in range(len(_split) + 1)]
    sub_fitness = [[] for _ in range(len(_split) + 1)]
    for i, (ind, objval) in enumerate(zip(population, objVals)):
        is_classified = False
        for j, ref_ in enumerate(_split):
            if objval[0] < ref_:
                sub_populations[j].append(ind)
                sub_fitness[j].append(objval)
                is_classified = True
                break
        if not is_classified:
            sub_populations[-1].append(ind)
            sub_fitness[-1].append(objval)
    return np.asarray(sub_populations, dtype=object), np.asarray(sub_fitness, dtype=object)


def archive_updating(population, objVals, archive_pop_set,  archive_objVal_set):
    # split population
    sub_populations, sub_fitness = split_population(population, objVals)
    # non-dominated sorting and update archive set
    archive_pop_set = copy.deepcopy(archive_pop_set)
    archive_objVal_set = copy.deepcopy(archive_objVal_set)
    # obtain non-dominated inds in each group
    for i in range(len(sub_populations)):
        if len(archive_objVal_set[i]) == 0:
            total_pop = np.asarray(sub_populations[i])
            total_fitness = np.asarray(sub_fitness[i])
        elif len(sub_fitness[i])==0:
            total_pop = np.asarray(archive_pop_set[i])
            total_fitness = np.asarray(archive_objVal_set[i])
        else:
            total_pop = np.vstack((archive_pop_set[i], sub_populations[i]))
            total_fitness = np.vstack((archive_objVal_set[i], sub_fitness[i]))

            total_fitness, indices = np.unique(total_fitness, return_index=True, axis=0)
            total_pop = total_pop[indices]

        if not total_fitness.size == 0:
            FrontValue_1_index = Utils.NDSort(total_fitness, total_pop.shape[0])[0] == 1
            FrontValue_1_index = np.reshape(FrontValue_1_index, (-1,))
            archive_pop_set[i] = total_pop[FrontValue_1_index]
            archive_objVal_set[i] = total_fitness[FrontValue_1_index]

    total_archive_pop, total_archive_objVals = [], []
    for pop, objVal in zip(archive_pop_set, archive_objVal_set):
        if not np.asarray(pop).size == 0:
            total_archive_pop.append(pop)
            total_archive_objVals.append(objVal)
    total_archive_pop = np.vstack((total_archive_pop))
    total_archive_objVals = np.vstack((total_archive_objVals))

    if len(total_archive_pop) > params['pop_size']:
        # include the overall front1 inds first, and the remaining uses binary-tournament-selection
        overall_Front_1_index = Utils.NDSort(total_archive_objVals, total_archive_pop.shape[0])[0] == 1
        overall_Front_1_index = np.reshape(overall_Front_1_index, (-1,))
        new_population = copy.deepcopy(total_archive_pop[overall_Front_1_index])
        new_objVals = copy.deepcopy(total_archive_objVals[overall_Front_1_index])

        idxs = [idx for idx, m in enumerate(overall_Front_1_index) if m]
        total_archive_pop = np.delete(total_archive_pop, idxs, 0)
        total_archive_objVals = np.delete(total_archive_objVals, idxs, 0)

        selection = Selection()
        fitness_list = [calc_fitness(total_archive_objVals[:, 1][i]) for i in range(len(total_archive_objVals[:, 1]))]
        selected_index_list = selection.binary_tournament_selection(fitness_list, k=params['pop_size']-len(new_population))

        for i in selected_index_list:
            new_population = np.append(new_population, np.asarray([total_archive_pop[i]]), axis=0)
            new_objVals = np.append(new_objVals, np.asarray([total_archive_objVals[i]]), axis=0)

    elif len(total_archive_pop) < params['pop_size']:
        # collect the inds from the combined pop gradually by their fronts, then randomly move the extra inds out
        front = 1
        new_population, new_objVals = [], []
        first_time = True
        while len(new_objVals) < params['pop_size']:
            for i in range(len(sub_populations)):
                if len(archive_objVal_set[i])==0:
                    total_pop_i = np.asarray(sub_populations[i])
                    total_fitness_i = np.asarray(sub_fitness[i])
                elif len(sub_fitness[i])==0:
                    total_pop_i = np.asarray(archive_pop_set[i])
                    total_fitness_i = np.asarray(archive_objVal_set[i])
                else:
                    total_pop_i = np.vstack((archive_pop_set[i], sub_populations[i]))
                    total_fitness_i = np.vstack((archive_objVal_set[i], sub_fitness[i]))

                    total_fitness_i, indices = np.unique(total_fitness_i, return_index=True, axis=0)
                    total_pop_i = total_pop_i[indices]

                if not total_fitness_i.size == 0:
                    FrontValue_index = Utils.NDSort(total_fitness_i, total_pop_i.shape[0])[0] == front
                    FrontValue_index = np.reshape(FrontValue_index, (-1,))
                    if not FrontValue_index.size==0:
                        if front > 1:
                            new_population = np.append(new_population, copy.deepcopy(total_pop_i[FrontValue_index]), axis=0)
                            new_objVals = np.append(new_objVals, copy.deepcopy(total_fitness_i[FrontValue_index]), axis=0)
                        else:
                            if first_time:
                                new_population = copy.deepcopy(total_pop_i[FrontValue_index])
                                new_objVals = copy.deepcopy(total_fitness_i[FrontValue_index])
                            else:
                                new_population = np.append(new_population, copy.deepcopy(total_pop_i[FrontValue_index]), axis=0)
                                new_objVals = np.append(new_objVals, copy.deepcopy(total_fitness_i[FrontValue_index]), axis=0)
                        first_time = False

            new_objVals = np.vstack((new_objVals))
            front = front + 1

        del_index = np.random.choice(range(len(new_population)), len(new_population) - params['pop_size'])
        new_population = copy.deepcopy(np.delete(np.asarray(new_population), del_index, 0))
        new_objVals = copy.deepcopy(np.delete(np.asarray(new_objVals), del_index, 0))

    else:
        new_population = copy.deepcopy(total_archive_pop)
        new_objVals = copy.deepcopy(total_archive_objVals)
    return new_population, new_objVals, archive_pop_set, archive_objVal_set

def plot_fig(objVals, total_archive_objVals, i, descriptions=None):
    fig = plt.figure('Generation#' + str(i + 1))
    ax3 = fig.add_subplot(111)
    if obj.__contains__('p'):
        label = '#Params'
        ax3.set_xlim((0, 3e6))
    else:
        label = '#MAdds'
        ax3.set_xlim((0, 600e6))

    ax3.set_ylim((-2000, -1500))
    ax3.set_xlabel(label, fontsize=12.5)
    ax3.set_ylabel(r'-$s_{wot}$', fontsize=12.5)

    ax3.scatter(objVals[:, 0], objVals[:, 1], s=50, c='blue', marker="o", edgecolors='black', label='updated population')
    ax3.scatter(total_archive_objVals[:, 0], total_archive_objVals[:, 1], s=40, c='red', marker="p", edgecolors='black', alpha=1.0, label='archive')
    plt.legend()
    # plt.show()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    if descriptions:
        plt.savefig('./img_txt/' + str(i + 1) + '_' + str(descriptions) + '.png')
    else:
        plt.savefig('./img_txt/' + str(i + 1) + '.png')
    plt.clf()
    plt.close()

def evolveCNN(params):
    torch.cuda.set_device(args.gpu)

    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    population = initialize_population(params)
    archive_pop_set = [[] for _ in range(len(_split)+1)]
    archive_objVal_set = [[] for _ in range(len(_split) + 1)]

    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
    errs_parent, NumParams_parent, flops_parent, fitnessSet_parent = fitness_evaluate(population, gen_no)

    if obj == 'p':
        objVals = np.asarray([NumParams_parent, errs_parent]).T
    else:
        objVals = np.asarray([flops_parent, errs_parent]).T
    population = np.asarray(population, dtype=object)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))


    sub_populations, sub_fitness = split_population(population, objVals)
    for i in range(len(sub_populations)):
        total_pop_i = np.asarray(sub_populations[i])
        total_fitness_i = np.asarray(sub_fitness[i])
        if not total_fitness_i.size == 0:
            FrontValue_1_index = Utils.NDSort(total_fitness_i, total_pop_i.shape[0])[0] == 1
            FrontValue_1_index = np.reshape(FrontValue_1_index, (-1,))
            archive_pop_set[i] = total_pop_i[FrontValue_1_index]
            archive_objVal_set[i] = total_fitness_i[FrontValue_1_index]

    total_archive_pop,total_archive_objVals = [],[]
    for pop,objVal in zip(archive_pop_set, archive_objVal_set):
        if not np.asarray(pop).size == 0:
            total_archive_pop.append(pop)
            total_archive_objVals.append(objVal)

    total_archive_pop = np.vstack((total_archive_pop))
    total_archive_objVals = np.vstack((total_archive_objVals))
    Utils.save_population('population', population, errs_parent, NumParams_parent, flops_parent, gen_no)
    Utils.save_population('archive', list(map(list,total_archive_pop)), total_archive_objVals.T[1], total_archive_objVals.T[0], np.zeros(len(total_archive_objVals)), gen_no)
    plot_fig(objVals, total_archive_objVals, gen_no)
    gen_no += 1

    for curr_gen in range(gen_no, params['num_generations']):
        torch.cuda.empty_cache()
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin evolution' % (curr_gen))
        population = evolve(population, fitnessSet_parent, params)
        Log.info('EVOLVE[%d-gen]-Finish evolution' % (curr_gen))

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        errs_offspring, NumParames_offspring, flops_offspring, fitnessSet_offspring = fitness_evaluate(population, curr_gen)

        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))
        if obj == 'p':
            objVals = np.asarray([NumParames_offspring, errs_offspring]).T
        else:
            objVals = np.asarray([flops_offspring, errs_offspring]).T
        population = np.asarray(population, dtype=object)

        population, objVals, archive_pop_set, archive_objVal_set = archive_updating(population, objVals, archive_pop_set, archive_objVal_set)
        fitnessSet_parent = [calc_fitness(objVals[:, 1][i]) for i in range(len(objVals[:, 1]))]

        total_archive_pop, total_archive_objVals = [], []
        for pop, objVal in zip(archive_pop_set, archive_objVal_set):
            if not np.asarray(pop).size == 0:
                total_archive_pop.append(pop)
                total_archive_objVals.append(objVal)

        total_archive_pop = np.vstack((total_archive_pop))
        total_archive_objVals = np.vstack((total_archive_objVals))
        Utils.save_population('population', population, objVals[:, 1], objVals[:, 0], np.zeros(len(population)), curr_gen)
        Utils.save_population('archive', list(map(list, total_archive_pop)), total_archive_objVals.T[1], total_archive_objVals.T[0], np.zeros(len(total_archive_objVals)), curr_gen)
        plot_fig(objVals, total_archive_objVals, curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end-start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    overall_Front_1_index = Utils.NDSort(total_archive_objVals, total_archive_pop.shape[0])[0] == 1
    overall_Front_1_index = np.reshape(overall_Front_1_index, (-1,))
    final_individuals = copy.deepcopy(total_archive_pop[overall_Front_1_index])
    final_objVals = copy.deepcopy(total_archive_objVals[overall_Front_1_index])

    plot_fig(total_archive_objVals, final_objVals, params['num_generations'])
    Utils.save_population('archive', list(map(list, final_individuals)), final_objVals.T[1], final_objVals.T[0], np.zeros(len(final_objVals)), params['num_generations'])

    #########################################################
    # select the individual with lowest (-score) from each group for final performance test
    sub_populations, sub_fitness = split_population(final_individuals, final_objVals)
    selected_individuals, selected_objVals = [], []
    for i in range(len(sub_populations)):
        if sub_fitness[i]:
            idx_min = np.argmin(np.transpose(sub_fitness[i])[1])
            selected_objVals.append(sub_fitness[i][idx_min])
            selected_individuals.append(sub_populations[i][idx_min])

    plot_fig(final_objVals, np.asarray(selected_objVals), params['num_generations'], 'final_selected')
    Utils.save_population('final_selected', selected_individuals, np.transpose(selected_objVals)[1], np.transpose(selected_objVals)[0], np.zeros(len(final_objVals)), params['num_generations'])


    for idx,final_ind in enumerate(selected_individuals):
        Utils.generate_pytorch_file(final_ind, -1, -idx, args, is_test=True)
    ######################################################

def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)

if __name__ == '__main__':
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.empty_cache()

    create_directory()
    params = Utils.get_init_params()

    dataset = str(__read_ini_file('DATA', 'dataset'))
    args.dataset = dataset

    obj = str(__read_ini_file('SEARCH', 'obj'))
    if obj.__contains__('p'):
        _split = [0.5e6, 1e6, 1.5e6, 2e6]
    else:
        _split = [200e6, 300e6, 400e6, 500e6]

    if args.test:
        fitnessTest(args.script_name, -1, is_test=True, args=args)
    else:
        evolveCNN(params)

