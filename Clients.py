from torch import nn, optim, poisson
import torch
import copy
import logging
from threading import Thread
from utils import get_data
from torch.utils.data import DataLoader
import threading

###############################################
from skimage.util import random_noise
import numpy as np
###############################################


class Client:
    def __init__(self, config):
        self.config = config
        self.num = self.config.num_clients
        self.client_id = [i for i in range(0, self.num)]
        self.model = None
        self.local_model = None
        self.dataloaders = []
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []
        self.len_dataset = []
        self.test_acc = []

    def load_data(self):
        self.trainset, self.testset = get_data(self.config.dataset, self.config)
        for subset in self.trainset:
            loader = DataLoader(subset, batch_size=self.config.local_bs)
            self.dataloaders.append(loader)

    def clients_to_server(self):
        return self.client_id

    def get_model(self, model):
        self.model = model
        self.local_model = model

    def local_train(self, user_id, dataloaders, poison=False, verbose=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr)

        for e in range(self.config.local_ep):
            running_loss = 0
            running_corrects = 0
            epoch_loss = 0
            epoch_acc = 0

            for inputs, labels in dataloaders:

                if poison:
                    inputs = random_noise(inputs, mode='gaussian', mean=0, var=self.config.std ** 2, clip=True)
                    inputs = random_noise(inputs, mode='s&p', amount=self.config.amount)
                    inputs = torch.tensor(inputs, dtype=torch.float32)

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = int(running_corrects) / len(dataloaders.dataset)

            logging.debug('User {}: {} Loss: {:.4f} Acc: {:.4f}'.format(user_id, "training", epoch_loss, epoch_acc))
        # need be locked
        lock = threading.Lock()
        lock.acquire()
        self.weights.append(copy.deepcopy(model.state_dict()))
        self.epoch_loss.append(epoch_loss)
        self.running_corrects.append(int(running_corrects))
        self.len_dataset.append(len(dataloaders.dataset))
        # acc_, avg_loss_ = self.test()
        # self.test_acc.append(acc_)
        lock.release()

    def upload(self, info):
        return info

    def update(self, glob_weights):
        self.model.load_state_dict(glob_weights)

    def train(self, selected_client, mal_data_clients, mal_model_clients):
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []
        self.test_acc = []

        self.len_dataset = []

        # multithreading
        # threads = [Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client])) for client in
        #            selected_client]

        threads = []

        if len(mal_data_clients) == 0:
            for client in selected_client:
                threads.append(Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client])))
        else:
            for client in selected_client:
                if client not in mal_data_clients:
                    threads.append(Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client])))
                else:
                    threads.append(Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client], poison=True)))


        [t.start() for t in threads]
        [t.join() for t in threads]

        # if malicious_client != -1:
        #     print("INIT", malicious_client)
        #     # 악의적인 client -> 모델 초기화
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     tmp = copy.deepcopy(self.model)
        #     tmp.to(device)
        #     tmp.load_state_dict(self.weights[malicious_client])
        #     tmp.apply(init_weight)
        #     self.weights[malicious_client] = tmp.state_dict()
        
        # implement model poisoning by tweaking the weights of the malicious client model
        if len(mal_model_clients) != 0:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tmp = copy.deepcopy(self.model)
            tmp.to(device)

            for client in mal_model_clients:
                tmp.load_state_dict(self.weights[client])
                tmp.apply(tweak_weights)
                self.weights[client] = tmp.state_dict()


        threads = [Thread(target=self.test_local(user_id=client)) for client in selected_client]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # training details
        info = {"weights": self.weights, "loss": self.epoch_loss, "corrects": self.running_corrects,
                'len': self.len_dataset, "id": self.client_id, "test_acc": self.test_acc}
        return self.upload(info)

    def test_local(self, user_id):
        corrects = 0
        test_loss = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_model.load_state_dict(self.weights[user_id])
        model = copy.deepcopy(self.local_model)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.testset, batch_size=32, shuffle=True)
        for batch_id, (inputs, labels) in enumerate(dataloader):
            loss = 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        acc = int(corrects) / len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader.dataset)
        # print(corrects)
        # print(len(dataloader.dataset))
        # print(f"test_acc:{acc}",)

        lock = threading.Lock()
        lock.acquire()
        self.test_acc.append(acc)
        lock.release()

    def test(self):
        corrects = 0
        test_loss = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.testset, batch_size=32, shuffle=True)
        for batch_id, (inputs, labels) in enumerate(dataloader):
            loss = 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        acc = int(corrects) / len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader.dataset)
        # print(corrects)
        # print(len(dataloader.dataset))
        # print(f"test_acc:{acc}",)
        return acc, avg_loss


def init_weight(module):
    class_name = module.__class__.__name__

    if class_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant(module.bias.data, 0.0)


def tweak_weights(module):

    class_name = module.__class__.__name__

    if class_name.find("Conv") != -1 or class_name.find("Linear") != -1:
        mean = torch.mean(module.weight.data)
        module.weight.data.add_(mean)
        # module.weight.data.mul_(-1.5)
        module.bias.data.add_(mean)
        # module.bias.data.mul_(-1.5)