#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser()

parser.add_argument(
    '--results-directory',
    type=str, required=True,
    help='Directory to do robustness experiment within'
)
parser.add_argument(
    '--title',
    type=str, default='',
    help='Title of plot'
)
parser.add_argument(
    '--drop-trials',
    type=int, default=20,
    help='Number of module dropout simulations to run per N'
)
parser.add_argument(
    '--ylim', nargs='+',
    type=float, default=[0.0, 1.0],
    help='ylim for matplotlib'
)
parser.add_argument(
    '--show', default=False,
    action='store_true',
    help='Show the plot instead of saving'
)


def robustness_simulation(script_args):
    seeds = list(filter(lambda x: not x.startswith('plot') and \
                                  not x.endswith('.npz') and \
                                  not x.endswith('.pdf'),
                        os.listdir(script_args.results_directory)))

    all_lambdas = []

    for seed in seeds:
        with open(os.path.join(script_args.results_directory, seed, 'args.json')) as f:
            args = json.load(f)

        lambdas = args['lambda_values']
        all_lambdas += lambdas
        trials = args['n_trials']
        M = args['n_modules']

    all_lambdas = sorted(all_lambdas)

    step = max(M // 20, 1)
    Ns = list(range(1, M+1, step))
    if M not in Ns:
        Ns = Ns + [M]

    p_min = 1 - (1 - 0.999**(1/trials))**(1/M)
    ps = np.linspace(p_min, 1.0, 20)

    results = np.zeros((trials, script_args.drop_trials, len(all_lambdas), len(Ns)))
    results2 = np.zeros((trials, script_args.drop_trials, len(all_lambdas), len(ps)))

    pbar = tqdm(total=len(seeds) * trials * len(lambdas))
    for seed in seeds:
        with open(os.path.join(script_args.results_directory, seed, 'args.json')) as f:
            args = json.load(f)

        lambdas = args['lambda_values']

        for trial_i in range(trials):
            trial_dir = os.path.join(script_args.results_directory, seed, 'trial_{}'.format(trial_i))

            for lambda_ in lambdas:
                pbar.update(1)
                lambda_dir = os.path.join(trial_dir, str(lambda_))
                lambda_i = all_lambdas.index(lambda_)

                label_dir = os.path.join(lambda_dir, 'module_outputs', 'early_stop_test_labels.npy')
                mod_out_dir = os.path.join(lambda_dir, 'module_outputs', 'early_stop_test_outputs.npy')

                labels = np.load(label_dir)
                mod_outs = np.load(mod_out_dir)

                for N_i, N in enumerate(Ns):
                    for trial2_i in range(script_args.drop_trials):
                        idx = np.random.choice(M, N, replace=False)
                        subsample = np.mean(mod_outs[idx, ...], axis=0)
                        predictions = np.argmax(subsample, axis=1)
                        error = np.mean(predictions != labels)
                        results[trial_i, trial2_i, lambda_i, N_i] = error

                for p_i, p in enumerate(ps):
                    for trial2_i in range(script_args.drop_trials):
                        idx = np.random.random(size=M) < p
                        subsample = np.mean(mod_outs[idx, ...], axis=0)
                        predictions = np.argmax(subsample, axis=1)
                        error = np.mean(predictions != labels)
                        results2[trial_i, trial2_i, lambda_i, p_i] = error

    np.savez(os.path.join(script_args.results_directory, 'robustness'),
            robustness_data_N=results,
            robustness_data_p=results2,
            Ns=Ns,
            ps=ps,
            lambdas=all_lambdas)


def plot_robustness(script_args):

    data = np.load(os.path.join(script_args.results_directory, 'robustness.npz'))
    robustness_data = data['robustness_data_N']
    lambdas = data['lambdas']
    Ns = np.array(data['Ns']) - 1
    trials = robustness_data.shape[0]
    robustness_data = np.mean(robustness_data, axis=1)
    mean = np.mean(robustness_data, axis=0)
    std_err = np.std(robustness_data, axis=0) / np.sqrt(trials)

    include_lambdas = [0.0, 0.99, 1.0]

    for lambda_i, lambda_ in enumerate(lambdas):
        if lambda_ not in include_lambdas:
            continue
        plt.errorbar(Ns, 1-mean[lambda_i, ::-1], label='λ={}'.format(lambda_), yerr=std_err[lambda_i, :])
    plt.xticks(Ns[::2])
    plt.xlabel('Modules dropped')
    plt.ylabel('Test error')
    plt.ylim(tuple(script_args.ylim))
    plt.title(script_args.title)
    plt.legend()
    if script_args.show:
        plt.show()
    else:
        plt.savefig(os.path.join(script_args.results_directory, 'robustness_plot_N.pdf'))

    plt.clf()

    robustness_data = data['robustness_data_p']
    lambdas = data['lambdas']
    ps = 1 - np.array(data['ps'])
    trials = robustness_data.shape[0]
    robustness_data = np.mean(robustness_data, axis=1)
    mean = np.mean(robustness_data, axis=0)
    std_err = np.std(robustness_data, axis=0) / np.sqrt(trials)

    include_lambdas = [0.0, 0.99, 1.0]

    for lambda_i, lambda_ in enumerate(lambdas):
        if lambda_ not in include_lambdas:
            continue
        plt.errorbar(ps, 1-mean[lambda_i], label='λ={}'.format(lambda_), yerr=std_err[lambda_i, :])
    plt.xlabel('Module drop probability')
    plt.ylabel('Test error')
    plt.ylim(tuple(script_args.ylim))
    plt.title(script_args.title)
    plt.legend()
    if script_args.show:
        plt.show()
    else:
        plt.savefig(os.path.join(script_args.results_directory, 'robustness_plot_p.pdf'))




if __name__ == "__main__":
    script_args = parser.parse_args()

    yes = ['yes', 'y']
    no = ['no', 'n', '']

    run_robustness = True
    if os.path.isfile(os.path.join(script_args.results_directory,
                                   'robustness.npz')):
        print('Robustness simulation has already been run. Re-run? y/[n]')
        while True:
            ans = input()
            if ans in no:
                run_robustness = False
                break
            elif ans in yes:
                break
            else:
                print('Answer yes or no. Re-run robustness simulation? y/[n]')

    if run_robustness:
        robustness_simulation(script_args)

    plot_robustness(script_args)
