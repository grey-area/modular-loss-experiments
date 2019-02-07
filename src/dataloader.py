from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
import sys


def STL10Wrapper(**kwargs):
    if kwargs['train']:
        split_str = 'train'
    else:
        split_str = 'test'

    del(kwargs['train'])

    kwargs['split'] = split_str

    return datasets.STL10(**kwargs)

def SVHNWrapper(**kwargs):
    if kwargs['train']:
        split_str = 'train'
    else:
        split_str = 'test'

    del(kwargs['train'])

    kwargs['split'] = split_str

    return datasets.SVHN(**kwargs)


datasets_dict = {
    'MNIST': {
        'dataset': datasets.MNIST,
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),
                                             (0.3081,))
                     ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 10
    },
    'Fashion-MNIST': {
        'dataset': datasets.FashionMNIST,
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,),
                                             (0.3530,))
                     ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 10
    },
    'EMNIST': {
        'dataset': datasets.EMNIST,
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1751,),
                                             (0.3332,))
                     ]),
        'dset_kwargs': {'split': 'balanced'},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 47
    },
    'CIFAR-10': {
        'dataset': datasets.CIFAR10,
        'train_transform':  transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2470, 0.2435, 0.2616))
                            ]),
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616))
                     ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 10
    },
    'CIFAR-100': {
        'dataset': datasets.CIFAR100,
        'train_transform':  transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                     (0.2673, 0.2564, 0.2762))
                            ]),
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4865, 0.4409),
                                             (0.2673, 0.2564, 0.2762))
                     ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 100
    },
    'SVHN': {
        'dataset': SVHNWrapper,
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4377, 0.4438, 0.4728),
                                             (0.1980, 0.2010, 0.1970))
                     ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 10
    },
    'STL10': {
        'dataset': STL10Wrapper,
        'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4467, 0.4398, 0.4066),
                                             (0.2603, 0.2566, 0.2713))
                     ]),
        'dset_kwargs': {},
        'val_size': 1000,
        'distribution': 'categorical',
        'input_shape': (3, 96, 96),
        'output_dim': 10
    }
}



def get_dataloaders(batch_size, trial_i, dataset='MNIST', augment=False, early_stop=False):

    data_dir = './data/{}'.format(dataset)

    params = datasets_dict[dataset]

    datasets = {}
    for split in ['train', 'valid', 'test']:
        if augment and split=='train' and 'train_transform' in params.keys():
            transform = params['train_transform']
        else:
            transform = params['transform']

        dset = params['dataset'](root=data_dir,
                                 train=(split != 'test'),
                                 download=True,
                                 transform=transform,
                                 **params['dset_kwargs'])
        datasets[split] = dset

    # Deterministic train/val split based on trial number
    indices = list(range(len(datasets['train'])))
    val_size = params['val_size']

    s = np.random.RandomState(trial_i)
    valid_idx = s.choice(indices, size=val_size, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    default_dloader_args = {
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': 4,
        'drop_last': False
    }

    dataloaders = {}

    # If we're not doing early stopping, don't use a separate validation set
    if early_stop:
        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                          sampler=train_sampler,
                                          **default_dloader_args)
        dataloaders['valid'] = DataLoader(dataset=datasets['valid'],
                                          sampler=valid_sampler,
                                          **default_dloader_args)
        dataloaders['test'] = DataLoader(dataset=datasets['test'],
                                         shuffle=False,
                                         **default_dloader_args)
    else:
        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                          shuffle=True,
                                          **default_dloader_args)
        dataloaders['valid'] = DataLoader(dataset=datasets['test'],
                                          shuffle=False,
                                          **default_dloader_args)
        dataloaders['test'] = DataLoader(dataset=datasets['test'],
                                         shuffle=False,
                                         **default_dloader_args)

    return dataloaders, params
