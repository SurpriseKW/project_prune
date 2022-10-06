from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def get_cifar10(batch_size=16, num_workers=0):
    trainLoader = DataLoader(
        CIFAR10(
            'fun/datasets',
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            download=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    testLoader = DataLoader(
        CIFAR10(
            'fun/datasets',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            download=True
        ),
        batch_size=batch_size,
        num_workers=num_workers
    )
    return trainLoader, testLoader