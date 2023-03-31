import pandas as pd
import torch

from simulate.simulation import Simulator
from util.option import args_parser
import matplotlib.pyplot as plt


def multiple_simulate():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    strategies = ['fed_avg', 'krum', 'median', 'trimmed_mean', 'sided_discard', 'ckks_sided_discard']
    for index, agg_strategy in enumerate(strategies):
        args.strategy = agg_strategy
        if index == len(strategies) - 1:
            simulator = Simulator(args, True)
        else:
            simulator = Simulator(args)
        simulator.start()


if __name__ == '__main__':
    # multiple_simulate()
    file_path = './result/mnist_iid_clients_20_epochs_20_malicious_0.3.csv'
    df = pd.read_csv(file_path, index_col='epochs')
    df.iloc[:, 1:6].plot()
    plt.show()
