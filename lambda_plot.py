#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
    '--show', default=False,
    action='store_true',
    help='Show the plot instead of saving'
)

script_args = parser.parse_args()


seeds = list(filter(lambda x: not x.startswith('plot') and \
                              not x.endswith('pdf') and \
                              not x.startswith('robustness') and \
                              not x.endswith('.py') and \
                              not x.endswith('~'),
                    os.listdir(script_args.results_directory)))

all_lambdas = []

for seed in seeds:
    with open(os.path.join(script_args.results_directory, seed, 'args.json')) as f:
        args = json.load(f)

    lambdas = args['lambda_values']
    all_lambdas += lambdas
    epochs = args['epochs']
    trials = args['n_trials']

all_lambdas = sorted(all_lambdas)
val_results = np.zeros((trials, len(all_lambdas), epochs, 2, 2))
test_results = np.zeros((trials, len(all_lambdas)))

for seed in seeds:
    seed_dir = os.path.join(script_args.results_directory, seed)

    with open(os.path.join(seed_dir, 'args.json')) as f:
        args = json.load(f)

    lambdas = args['lambda_values']

    for trial_i in range(trials):
        trial_dir = os.path.join(seed_dir, 'trial_{}'.format(trial_i))

        for lambda_ in lambdas:
            lambda_dir = os.path.join(trial_dir, str(lambda_))
            lambda_i = all_lambdas.index(lambda_)

            train_val_data = np.load(os.path.join(lambda_dir, 'train_val_results.npy'))


            val_results[trial_i, lambda_i, :, 0, 0] = train_val_data[:, 0, 0]
            val_results[trial_i, lambda_i, :, 0, 1] = train_val_data[:, 0, 2]
            val_results[trial_i, lambda_i, :, 1, 0] = train_val_data[:, 1, 0]
            val_results[trial_i, lambda_i, :, 1, 1] = train_val_data[:, 1, 2]
            test_results[trial_i, lambda_i] = np.load(os.path.join(lambda_dir, 'test_results.npy'))[0, 2]


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

plot_i = 0

mean = np.percentile(val_results, 50, axis=0)
std = np.std(val_results, axis=0) / np.sqrt(trials)

splits = ['Training', 'Validation']
metrics = ['loss', 'error']

xs = range(epochs)
for split_i in [0, 1]:
    for metric_i in [0, 1]:
        for lambda_i, lambda_ in enumerate(all_lambdas):
            axes[plot_i // 3][plot_i % 3].plot(xs, mean[lambda_i, :, split_i, metric_i], c='C{}'.format(lambda_i), label='λ={}'.format(lambda_))
        axes[plot_i // 3][plot_i % 3].set_xlabel('Epochs')
        axes[plot_i // 3][plot_i % 3].set_ylabel('{} {}'.format(splits[split_i], metrics[metric_i]))
        plot_i += 1

axes[0][0].legend()


mean = np.mean(test_results, axis=0)
median = np.percentile(test_results, 50, axis=0)
std = np.std(test_results, axis=0) / np.sqrt(trials)
axes[plot_i // 3][plot_i % 3].errorbar(all_lambdas, mean, yerr=std)
axes[plot_i // 3][plot_i % 3].set_xlabel('λ')
axes[plot_i // 3][plot_i % 3].set_ylabel('Test error (mean)')

plot_i += 1

axes[plot_i // 3][plot_i % 3].errorbar(all_lambdas, median)
axes[plot_i // 3][plot_i % 3].set_xlabel('λ')
axes[plot_i // 3][plot_i % 3].set_ylabel('Test error (median)')

fig.suptitle(script_args.title)

if script_args.show:
    plt.show()
else:
    plt.savefig(os.path.join(script_args.results_directory, 'lambda_plots.pdf'))
