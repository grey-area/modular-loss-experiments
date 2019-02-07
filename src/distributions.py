import torch
import torch.nn as nn

distributions = {
    'gaussian': torch.nn.MSELoss(),
    'categorical': torch.nn.CrossEntropyLoss(),
    'poisson': torch.nn.PoissonNLLLoss()
}
