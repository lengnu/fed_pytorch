from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms

from util.constant import CIFAR10_DATA_PATH, MNIST_DATA_PATH


def mnist():
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = MNIST(root=MNIST_DATA_PATH, train=True, download=True, transform=trans_mnist)
    dataset_test = MNIST(root=MNIST_DATA_PATH, train=False, download=True, transform=trans_mnist)
    num_labels = 10
    return dataset_train, dataset_test, num_labels


def cifar10():
    trans_cifar10 = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(root=CIFAR10_DATA_PATH, train=True, download=True, transform=trans_cifar10)
    dataset_test = CIFAR10(root=CIFAR10_DATA_PATH, train=False, download=True, transform=trans_cifar10)
    num_labels = 10
    return dataset_train, dataset_test, num_labels
