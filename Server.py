import torch
import logging
from models import *
from Clients import Client
import copy
import numpy as np
import random
import torch.nn as nn


class Server:
    def __init__(self, args, file_logger):
        self.config = args
        self.model = self.load_model()
        self.clients = None
        self.client_index = []
        self.target_round = -1
        self.file_logger = file_logger

    def run(self):
        random_round = [2, 5, 9, 15, 21, 27]  # random.randint(1, self.config.rounds)

        self.connect_clients()
        for round in (range(1, self.config.rounds + 1)):
            logging.info("-" * 22 + "round {}".format(round) + "-" * 30)
            # select clients which participate training
            selected = self.clients_selection()
            logging.info("selected clients:{}".format(selected))
            random_client = -1
            # if random_round == round:          # I commented out this because this is wrong as it compares a list with an int which will always be false
            if round in random_round:            # This is the correct way to check if this specific round is in the random_round list
                random_client = random.randint(0, self.config.num_clients * self.config.frac - 1)
                print(random_client)
            info = self.clients.train(selected, random_client)

            logging.info("aggregate weights")
            # update glob model
            glob_weights = self.fed_avg(info)
            self.model.load_state_dict(glob_weights)

            train_acc = self.getacc(info)
            test_acc, test_loss = self.test()

            for i in range(len(info["test_acc"])):
                # id_round_weight_localLoss_localTest_test_maliciousClientId
                self.file_logger.debug(
                    "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(i, round, info["weights"][i], info["loss"][i],
                                                     info["test_acc"][i], test_acc, random_client))
            logging.info(
                "training acc: {:.4f}, test acc: {:.4f}, test_loss: {:.4f}\n".format(train_acc, test_acc, test_loss,
                                                                                    info["test_acc"]))
            if test_acc > self.config.target_accuracy:
                self.target_round = round
                logging.info("target achieved")
                break

            # broadcast glob weights
            self.clients.update(glob_weights)

    def load_model(self):
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        # Set up global model
        model = get_model(dataset, self.config)
        logging.debug(model)
        return model

    def fed_avg(self, info):
        weights = info["weights"]
        length = info["len"]
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        return w_avg

    def connect_clients(self):
        self.clients = Client(self.config)
        self.client_index = self.clients.clients_to_server()
        self.clients.get_model(self.model)
        self.clients.load_data()

    def clients_selection(self):
        # randomly selection
        frac = self.config.frac
        n_clients = max(1, int(self.config.num_clients * frac))
        training_clients = np.arange(n_clients)
        # training_clients = np.random.choice(self.client_index, n_clients, replace=False)
        return training_clients

    def test(self):
        return self.clients.test()

    def getacc(self, info):
        corrects = sum(info["corrects"])
        total_samples = sum(info["len"])
        return corrects / total_samples
