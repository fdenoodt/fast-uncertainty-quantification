from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def data_loaders(h: dict, transformations: list):
    transform = transforms.Compose([transforms.ToTensor()] + transformations)
    mnist_train = MNIST(h['dataset_path'], train=True, download=True, transform=transform)
    mnist_val = MNIST(h['dataset_path'], train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=h['batch_size'], shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=h['batch_size'], shuffle=False)
    return train_loader, val_loader
