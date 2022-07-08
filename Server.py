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
        random_round = [i for i in range(1, self.config.rounds + 1)]  # random.randint(1, self.config.rounds)

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

            ################# timpany ############
            new_info = self.timpany(info)
            #####################################
            
            logging.info("aggregate weights")
            # update glob model
            # glob_weights = self.fed_avg(info)
            glob_weights = self.fed_avg(new_info)
            self.model.load_state_dict(glob_weights)    # global model update

            train_acc = self.getacc(info)
            test_acc, test_loss = self.test()       # global model test accuracy and loss

            for i in range(len(info["test_acc"])):
                # id_round_weight_localLoss_localTest_globaltest_maliciousClientId
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
            self.clients.update(glob_weights)      # update the local model with the new global model

    def load_model(self):
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        # Set up global model
        model = get_model(dataset, self.config)
        logging.debug(model)
        return model

    #####################################################################
    def timpany(self, info):
        # the accuracies of the client models on the global test dataset is already stored in the info dictionary,
        # so we can proceed to the evaluation of local model accuracy using control charts
        sum = 0
        acc = []
        for i in range(len(info["test_acc"])):
            acc.append(info["test_acc"][i])
        acc = np.array(acc)

        for a in acc:
            sum += a
        
        CL = sum / len(acc)
        self.file_logger.debug(f"CL: {CL}")
        std_dev = np.std(acc)

        alpha = 1       # can be 1, 2 or 3
        # UCL = CL + 1.96 * std_dev / np.sqrt(len(acc))  # suggested by GitHub Copilot
        UCL = CL + (alpha * std_dev)
        LCL = CL - (alpha * std_dev)

        new_info = {'weights': [], 'len': []}    # will contain only those client models' weights which have accuracy greater than or equal to CL
        for b in range(len(acc)):
            if acc[b] >= CL:
                new_info['weights'].append(info['weights'][b])
                new_info['len'].append(info['len'][b])
            
            else:
                self.file_logger.debug(f"client {b} has accuracy {info['test_acc'][b]}, and is rejected")
        
        return new_info
    
    #####################################################################

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
