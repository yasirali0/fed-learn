import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", type=str, default="INFO",
                        help='Log messages level.')
    parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                        help="the name of dataset")

    parser.add_argument("--num_clients", type=int, default=10)    # total number of clients in the federated learning system
    parser.add_argument("--frac", type=float, default=1)          # fraction of clients to be used for training in a round. values range from [0, 1]
    parser.add_argument("--mal_clients_frac", type=float, default=0.1)
    parser.add_argument("--mal_round_prob", type=float, default=0.5)
    parser.add_argument("--IID", type=bool, default=True)

    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--target_accuracy", default=0.99)
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--model", type=str, default="cifar-10")

    parser.add_argument("--file_name", type=str, default="test.log", help="the log file name")

    parser.add_argument("--data_v_model_poison", type=float, default=0.5)

    parser.add_argument("--std", type=float, help="standard deviation value for the gaussian noise to add to the training data of a malicious client", default=0.0)
    parser.add_argument("--amount", type=float, help="amount value for the salt&pepper noise to add to the training data of a malicious client. range is <0, 1>", default=0.0)

    parser.add_argument("--use_timpany", type=bool, default=True)

    args = parser.parse_args()

    return args
