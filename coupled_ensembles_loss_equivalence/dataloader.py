import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR-10', train=True,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

    return loader
