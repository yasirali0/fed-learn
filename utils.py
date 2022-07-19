from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from torch._utils import _accumulate
import random
import numpy as np


def get_data(dataset, config):
    if dataset == "cifar10":
        return CIFAR10(config).load_data(IID=config.IID)

    #################################################
    if dataset == "cifar100":
        return CIFAR100(config).load_data(IID=config.IID)
    
    if dataset == "MNIST":
        return MNIST(config).load_data(IID=config.data.IID)


class CIFAR10:
    def __init__(self, config):
        self.config = config
        self.trainset = None
        self.testset = None

    def load_data(self, IID=True):
        data_dir = './data/' + self.config.dataset + "/"
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.trainset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        self.testset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        total_clients = self.config.num_clients
        total_sample = self.trainset.data.shape[0]
        length = [total_sample // total_clients] * total_clients

        if IID:
            splited_train = random_split(self.trainset, length)

        return splited_train, self.testset


class CIFAR100:
    def __init__(self, config):
        self.config = config
        self.trainset = None
        self.testset = None

    def load_data(self, IID=True):
        data_dir = './data/' + self.config.dataset + "/"
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        self.trainset = datasets.CIFAR100(data_dir, train=True, download=True,
                                         transform=apply_transform)

        self.testset = datasets.CIFAR100(data_dir, train=False, download=True,
                                        transform=apply_transform)

        total_clients = self.config.num_clients
        total_sample = self.trainset.data.shape[0]
        length = [total_sample // total_clients] * total_clients

        if IID:
            splited_train = random_split(self.trainset, length)

        return splited_train, self.testset



class MNIST:
    def __init__(self, config):
        self.config = config
        print(self.config.paths.data)
        self.path = str(self.config.paths.data) + '/' + self.config.dataset
        self.trainset = None
        self.testset = None

    def load_data(self, IID=True):
        self.trainset = datasets.MNIST(
            self.path, train=True, download=True, transform=transforms.Compose([
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            self.path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        total_clients = self.config.clients.total
        total_sample = self.trainset.data.shape[0]
        # number of samples on each client
        length = [total_sample // total_clients] * total_clients
        if IID:
            spilted_train = random_split(self.trainset, length)

        else:
            print("None-IID")
            if sum(length) != len(self.trainset):
                raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
            index = []
            for i in range(10):
                index.append([])

            i = 0
            for img, label in self.trainset:
                index[label].append(i)
                i += 1

            indices = np.array([elem for c_list in index for elem in c_list]).reshape(-1, 200)

            np.random.shuffle(indices)
            indices = indices.flatten()
            print(indices.shape)

            spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
                             zip(_accumulate(length), length)]
            print(len(spilted_train))
        return spilted_train, self.testset
