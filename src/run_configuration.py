import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import json
import shutil
import numpy as np
import random
import logging

from argument_parser import parse_args
from dataloader import get_dataloaders
from distributions import distributions
from models import DenseModel, ConvModel, DenseNet


def do_batch(model, inputs, labels, distribution,
             args, lambda_, optimizer=None):

    criterion = distributions[distribution]

    eta_hat = model(inputs)
    eta_bar = torch.mean(eta_hat, dim=0)

    ensemble_loss = criterion(eta_bar, labels)
    module_losses = [criterion(eta_hat[i, ...], labels) for i in range(args.n_modules)]
    average_loss = torch.mean(torch.stack(module_losses))

    if optimizer is not None:
        optimizer.zero_grad()
        training_loss = lambda_ * ensemble_loss + (1 - lambda_) * average_loss
        training_loss.backward()
        optimizer.step()

    if distribution == 'categorical':
        predictions = torch.argmax(eta_bar.detach(), dim=1)
        error = 1 - np.mean((predictions==labels).cpu().numpy())
        module_predictions = [torch.argmax(eta_hat[i, ...].detach(), dim=1) for i in range(args.n_modules)]
        module_errors = [1 - np.mean((pred==labels).cpu().numpy()) for pred in module_predictions]
    else:
        error = -1.0
        module_errors = [-1.0] * args.n_modules

    ensemble_loss = ensemble_loss.detach().cpu().numpy()
    module_losses = [loss.detach().cpu().numpy() for loss in module_losses]
    average_loss = average_loss.detach().cpu().numpy()

    entropies = list(torch.mean(torch.sum(-nn.Softmax(dim=2)(eta_hat) * nn.LogSoftmax(dim=2)(eta_hat), dim=2), dim=1).detach().cpu().numpy())

    return np.array([ensemble_loss, average_loss, error] + module_losses + module_errors + entropies), eta_hat.detach().cpu().numpy()


def do_epoch(loader, device, model, args, distribution,
             lambda_, optimizer=None, get_predictions=False):

    running_totals = np.zeros(3 + 3 * args.n_modules)
    epoch_logits = np.zeros((args.n_modules, 0, model.output_dim))
    epoch_labels = np.zeros(0)

    for batch_i, (inputs, labels) in enumerate(loader):
        if get_predictions:
            epoch_labels = np.concatenate((epoch_labels, labels.numpy()),
                                          axis=0)
        inputs, labels = inputs.to(device), labels.to(device)

        metrics, logits = do_batch(model, inputs, labels, distribution,
                                   args, lambda_, optimizer)

        running_totals += metrics
        if get_predictions:
            epoch_logits = np.concatenate(
                (epoch_logits, logits),
                axis=1
            )

    epoch_labels = epoch_labels.astype(np.int64)

    return running_totals / (batch_i + 1), epoch_logits, epoch_labels


def do_test_set(name, lambda_dir, test_loader, device, model, args,
                distribution):
    model.train(mode=False)

    metrics, logits, labels = do_epoch(test_loader, device, model, args,
                                       distribution, 1.0,
                                       optimizer=None,
                                       get_predictions=True)
    np.save(os.path.join(lambda_dir, '{}_test_outputs'.format(name)), logits)
    np.save(os.path.join(lambda_dir, '{}_test_labels'.format(name)), labels)

    return metrics


def do_lambda_value(model, lambda_, learning_rate, args, loaders,
                    distribution, device, lambda_dir):
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.use_nesterov
    )
    optimizers = {'train': optimizer, 'valid': None}
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.learning_rate_decay_milestones,
                                               gamma=args.learning_rate_decay_factor)

    weight_dir = os.path.join(lambda_dir, 'temp_weights')
    os.makedirs(weight_dir, exist_ok=True)

    results = np.full((args.epochs, 2, 3 + 3 * args.n_modules), np.inf)

    min_epoch = 0
    min_val_error = np.inf
    epochs_since_improvement = 0

    weight_norms = None
    grad_norms = None

    for epoch in range(args.epochs):
        for split_i, split in enumerate(['train', 'valid']):

            # Switch between training and evaluation modes
            # e.g., for dropout or batch norm layers
            model.train(mode=(split=='train'))
            metrics, _, _ = do_epoch(loaders[split], device, model, args,
                                     distribution, lambda_,
                                     optimizer=optimizers[split])
            results[epoch, split_i, :] = metrics

            if split=='train':
                if grad_norms is None:
                    weight_norms, grad_norms = model.weight_grad_norms()
                else:
                    new_weight_norms, new_grad_norms = model.weight_grad_norms()
                    weight_norms = np.concatenate((weight_norms, new_weight_norms), axis=0)
                    grad_norms = np.concatenate((grad_norms, new_grad_norms), axis=0)

        # If we're doing classification, use the validation error for early
        # stopping. Else use the loss.
        if distribution == 'categorical':
            val_error = results[epoch, 1, 2]
        else:
            val_error = results[epoch, 1, 0]

        if val_error < min_val_error:
            min_val_error = val_error
            min_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        print(f'Epoch: {epoch}\tVal. error: {val_error:.02f}\tEpochs since improvement: {epochs_since_improvement}')

        weight_path = os.path.join(weight_dir,
                                   'epoch_{}_weights.pt'.format(epoch))
        torch.save(model.state_dict(), weight_path)
        scheduler.step()

    output_dir = os.path.join(lambda_dir, 'module_outputs')
    os.makedirs(output_dir, exist_ok=True)

    test_results = np.zeros((2, 3 + 3 * args.n_modules))

    # Get and save predictions on test set at final epoch
    test_results[1, :] = do_test_set('final', output_dir, loaders['test'],
                                     device, model, args, distribution)

    # Keep weights for the early stopping epoch and final epoch
    final_weight_dir = os.path.join(lambda_dir, 'weights')
    os.makedirs(final_weight_dir, exist_ok=True)

    shutil.copy(
        os.path.join(weight_dir, f'epoch_{args.epochs - 1}_weights.pt'),
        os.path.join(final_weight_dir, 'final_weights.pt')
    )
    if args.early_stop:
        shutil.copy(
            os.path.join(weight_dir, f'epoch_{min_epoch}_weights.pt'),
            os.path.join(final_weight_dir, 'early_stop_weights.pt')
        )

    # Re-load weights and then delete weight files
    if args.early_stop:
        reload_weight_dir = os.path.join(weight_dir,
                                         'epoch_{}_weights.pt'.format(min_epoch))
        print("Re-loading weights from epoch {}".format(min_epoch))
        print("Re-loading weights from {}".format(reload_weight_dir))
        model.load_state_dict(torch.load(reload_weight_dir))
    print("Deleting weight files")
    shutil.rmtree(weight_dir)

    # Save results file
    np.save(os.path.join(lambda_dir, 'train_val_results'), results)

    # Get and save predictions on test set at early stopping epoch
    if args.early_stop:
        test_results[0, :] = do_test_set("early_stop", output_dir, loaders['test'],
                                         device, model, args, distribution)
    np.save(os.path.join(lambda_dir, 'test_results'), test_results)
    np.save(os.path.join(lambda_dir, 'weight_norms'), weight_norms)
    np.save(os.path.join(lambda_dir, 'gradient_norms'), grad_norms)


def run(args, use_cuda, output_dir):

    trial_list = list(range(args.n_trials))
    np.random.shuffle(trial_list)

    for trial_i in trial_list:
        trial_dir = os.path.join(output_dir, 'trial_{}'.format(trial_i))
        os.makedirs(trial_dir, exist_ok=True)

        loaders, params = get_dataloaders(args.batch_size, trial_i,
                                          args.dataset, args.augment_data,
                                          early_stop=args.early_stop)

        if args.network_type == 'fc':
            model = DenseModel(input_dim=np.prod(params['input_shape']),
                               output_dim=params['output_dim'],
                               hidden_nodes=args.hidden_nodes,
                               num_modules=args.n_modules,
                               activation=args.activation)
        elif args.network_type == 'conv':
            model = ConvModel(input_shape=params['input_shape'],
                              output_dim=params['output_dim'],
                              num_filters=args.filters,
                              kernel_sizes=args.kernels,
                              strides=args.strides,
                              dilations=args.dilations,
                              num_modules=args.n_modules,
                              activation=args.activation,
                              final_layer=args.conv_final_layer)
        elif args.network_type == 'densenet':
            model = DenseNet(input_shape=params['input_shape'],
                             output_dim=params['output_dim'],
                             growth_rate=args.densenet_k,
                             depth=args.densenet_depth,
                             reduction=args.densenet_reduction,
                             bottleneck=args.densenet_bottleneck,
                             num_modules=args.n_modules)

        logging.debug(args)
        logging.debug('Parameters: {}'.format(model.n_parameters()))

        device = torch.device("cuda" if use_cuda else "cpu")
        model = model.to(device)
        model.reset_parameters()

        weight_path = os.path.join(trial_dir, 'initial_weights.pt')
        torch.save(model.state_dict(), weight_path)

        for lambda_i, (lambda_, learning_rate) in enumerate(
                zip(args.lambda_values, args.learning_rates)):
            model.load_state_dict(torch.load(weight_path))

            lambda_dir = os.path.join(trial_dir, str(lambda_))
            os.makedirs(lambda_dir, exist_ok=True)

            do_lambda_value(model, lambda_, learning_rate, args, loaders,
                            params['distribution'], device, lambda_dir)


def initialize():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    assert len(args.learning_rates) == 1 or \
           len(args.learning_rates) == len(args.lambda_values), \
           'Learning rates list should be length 1 or equal to lambda list length'
    if len(args.learning_rates) == 1:
        args.learning_rates *= len(args.lambda_values)

    args.lambda_values, args.learning_rates = zip(*sorted(zip(args.lambda_values, args.learning_rates)))

    assert (min(args.lambda_values) >= 0.0 and max(args.lambda_values) <= 1.0),\
           'Lambda values outside of [0, 1]'
    use_cuda = torch.cuda.is_available() and not args.cpu

    output_dir = os.path.join(
        'results', args.output_directory, str(args.seed)
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    return args, use_cuda, output_dir


def main():
    args, use_cuda, output_dir = initialize()

    run(args, use_cuda, output_dir)


if __name__ == "__main__":
    main()
