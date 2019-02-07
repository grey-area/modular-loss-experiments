import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from arg_parser import parse_args
from dataloader import get_train_dataloader
from model import Net


def common_init(args, data):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    nets = [Net() for i in range(args.num_modules)]
    logits_list = [net(data) for net in nets]

    return nets, logits_list


def get_gradients(nets):
    return torch.cat([torch.cat([(p.grad.data).view(p.numel()) for p in net.parameters()]) for net in nets])


def get_FC_grads(args, data, labels):
    nets, logits_list = common_init(args, data)

    average_logits = sum(logits_list) / args.num_modules
    loss = nn.CrossEntropyLoss()(average_logits, labels)
    loss.backward()

    return get_gradients(nets)


def get_SM_grads(args, data, labels):
    nets, logits_list = common_init(args, data)

    average_log_softmax = sum([nn.LogSoftmax(dim=1)(logits) for logits in logits_list]) / args.num_modules
    loss = nn.NLLLoss()(average_log_softmax, labels)
    loss.backward()

    return get_gradients(nets)


def get_LL_grads(args, data, labels):
    nets, logits_list = common_init(args, data)

    loss = sum([nn.CrossEntropyLoss()(logits, labels) for logits in logits_list]) / args.num_modules
    loss.backward()

    return get_gradients(nets)


def get_independent_grads(args, data, labels):
    nets, logits_list = common_init(args, data)

    losses = [nn.CrossEntropyLoss()(logits, labels) for logits in logits_list]
    for loss in losses:
        loss.backward()

    return get_gradients(nets)


if __name__ == "__main__":
    args = parse_args()
    data, labels = next(iter(get_train_dataloader(args.batch_size)))

    print(f'\nDemonstrating equivalence of LL, SM, and independent training methods with e={args.num_modules} networks')

    FC_grads = get_FC_grads(args, data, labels)
    SM_grads = get_SM_grads(args, data, labels)
    LL_grads = get_LL_grads(args, data, labels)
    independent_grads = get_independent_grads(args, data, labels)

    print('\nRatios of parameter gradients with LL and SM methods:')
    LL_SM_ratios = LL_grads / SM_grads
    print(LL_SM_ratios)

    print('\nRatios of parameter gradients with independent and SM methods:')
    independent_SM_ratios = independent_grads / SM_grads
    print(independent_SM_ratios)

    print('\nRatios of parameter gradients with FC and SM methods:')
    FC_SM_grads = FC_grads / SM_grads
    print(FC_SM_grads)
