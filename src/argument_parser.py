import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n-trials',
        type=int, default=1,
        help='Number of trials'
    )

    parser.add_argument(
        '--n-modules',
        type=int, default=1,
        help='Number of modules'
    )

    parser.add_argument(
        '--early-stop', default=False,
        action='store_true', help='Whether to use a validation set for early stopping'
    )

    parser.add_argument(
        '--dataset', type=str,
        default='MNIST',
        choices=['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10', 'CIFAR-100',
                 'SVHN', 'STL10'],
        help='Dataset to use'
    )

    parser.add_argument(
        '--augment-data', default=False,
        action='store_true', help='Whether to augment training data'
    )

    parser.add_argument(
        '--activation', type=str,
        default='relu',
        choices=['relu', 'sigmoid'],
        help='Activation function'
    )

    parser.add_argument(
        '--network-type', type=str,
        default='fc',
        choices=['fc', 'conv', 'densenet'],
        help='Type of neural network'
    )

    parser.add_argument(
        '--hidden-nodes', nargs='+',
        type=int, default=[5],
        help='List of number of hidden nodes per layer of FC network'
    )

    parser.add_argument(
        '--conv-final-layer', type=str,
        default='avg',
        choices=['avg', 'fc'],
        help='Specifies whether the convolutional part of the network is ' + \
             'followed by average pooling or a fully connected layer'
    )

    parser.add_argument(
        '--filters', nargs='+',
        type=int, default=[5],
        help='List of number of filters per layer of convolutional network'
    )

    parser.add_argument(
        '--kernels', nargs='+',
        type=int, default=[3],
        help='List of kernel sizes per layer of convolutional network. ' + \
             'Must be of length 1 or equal the length of the filters arg'
    )

    parser.add_argument(
        '--strides', nargs='+',
        type=int, default=[1],
        help='List of strides per layer of convolutional network. ' + \
             'Must be of length 1 or equal the length of the filters arg'
    )

    parser.add_argument(
        '--dilations', nargs='+',
        type=int, default=[1],
        help='List of dilations per layer of convolutional network. ' + \
             'Must be of length 1 or equal the length of the filters arg'
    )

    parser.add_argument(
        '--densenet-depth', type=int,
        default=100, help='DenseNet depth. Should be 3n+4 for some n > 1'
    )

    parser.add_argument(
        '--densenet-k', type=int,
        default=12, help='DenseNet growth rate'
    )

    parser.add_argument(
        '--densenet-reduction', type=float,
        default=0.5, help='DenseNet reduction factor. Between 0 and 1.'
    )

    parser.add_argument(
        '--densenet-bottleneck', default=False,
        action='store_true', help='Use bottleneck layers in DenseNet'
    )

    parser.add_argument(
        '--lambda-values', nargs='+',
        type=float, required=True,
        help='List of lambda values to use'
    )

    parser.add_argument(
        '--batch-size', type=int,
        default=100, help='Batch size'
    )

    parser.add_argument(
        '--epochs', type=int,
        default=1, help='Number of epochs'
    )

    parser.add_argument(
        '--learning-rates', nargs='+',
        type=float, default=[0.02],
        help='Learning rate for SGD. Single value or one per lambda value'
    )

    parser.add_argument(
        '--learning-rate-decay-milestones', nargs='+',
        type=int, default=[],
        help='Sorted list of epochs at which to decay learning rate. Default = [], no learning rate decay.'
    )

    parser.add_argument(
        '--learning-rate-decay-factor', type=float,
        default='0.1', help='Learning rate decay factor'
    )

    parser.add_argument(
        '--weight-decay', type=float,
        default='0.0', help='Weight decay'
    )

    parser.add_argument(
        '--momentum', type=float,
        default='0.9', help='Momentum for SGD'
    )

    parser.add_argument(
        '--use-nesterov', default=False,
        action='store_true', help='Use Nesterov momentum'
    )

    parser.add_argument(
        '--seed', type=int,
        default=1230, help='Seed for Torch'
    )

    parser.add_argument(
        '--cpu', default=False,
        action='store_true', help="Don't use CUDA"
    )

    parser.add_argument(
        '--debug', default=False,
        action='store_true', help='Print layer output shapes then exit'
    )

    parser.add_argument(
        '--output-directory', type=str,
        required=True, help='Directory where results are stored'
    )

    return parser.parse_args()
