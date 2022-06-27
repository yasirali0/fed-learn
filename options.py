import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", type=str, default="INFO",
                        help='Log messages level.')
    parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                        help="the name of dataset")

    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--frac", type=float, default=1)          # fraction of clients to be used for training in a round. [0, 1]
    parser.add_argument("--per_round", type=int, default=10)
    parser.add_argument("--IID", type=bool, default=True)

    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--target_accuracy", default=0.99)
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--model", type=str, default="cifar-10")

    parser.add_argument("--file_name", type=str, default="test.log", help="the log file name")

    parser.add_argument("--data_poison", type=bool, default=False)

    args = parser.parse_args()

    return args
