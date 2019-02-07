import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-size', type=int,
        default=100, help='Batch size'
    )

    parser.add_argument(
        '--num-modules', type=int,
        default=4, help='Number of modules'
    )

    parser.add_argument(
        '--learning-rate', type=float,
        default=0.01, help='Learning rate for SGD'
    )

    parser.add_argument(
        '--momentum', type=float,
        default='0.9', help='Momentum for SGD'
    )

    parser.add_argument(
        '--seed', type=int,
        default=1230, help='Seed for Torch'
    )

    return parser.parse_args()
