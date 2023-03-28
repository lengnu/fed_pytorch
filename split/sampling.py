import os
import numpy as np
from fedlab.utils.dataset import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report
from util.constant import MNIST_SPLIT_PATH, CIFAR10_SPLIT_PATH


def cifar10_balance_iid(dataset, num_clients: int):
    partition_info_path = CIFAR10_SPLIT_PATH + '/cifar10_split_iid_{:d}'.format(num_clients)
    if os.path.exists(partition_info_path):
        return np.load(partition_info_path + '/dict.npy', allow_pickle=True).item()
    else:
        os.makedirs(partition_info_path)
        balance_iid_part = CIFAR10Partitioner(
            dataset.targets,
            balance=True,
            num_clients=num_clients,
            partition='iid',
            verbose=False)
        np.save(partition_info_path + '/dict.npy', balance_iid_part.client_dict)
        partition_report(dataset.targets, balance_iid_part.client_dict,
                         class_num=10, verbose=False, file=partition_info_path + '/report.csv')
        return balance_iid_part.client_dict


def mnist_balance_iid(dataset, num_clients: int):
    partition_info_path = MNIST_SPLIT_PATH + '/mnist_split_iid_{:d}'.format(num_clients)
    if os.path.exists(partition_info_path):
        return np.load(partition_info_path + '/dict.npy', allow_pickle=True).item()
    else:
        os.makedirs(partition_info_path)
        balance_iid_part = MNISTPartitioner(
            dataset.targets,
            num_clients=num_clients,
            partition='iid',
            verbose=False)
        np.save(partition_info_path + '/dict.npy', balance_iid_part.client_dict)
        partition_report(dataset.targets, balance_iid_part.client_dict,
                         class_num=10, verbose=False, file=partition_info_path + '/report.csv')
        return balance_iid_part.client_dict
