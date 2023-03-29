from typing import List

import torch

from entity.abstract import AbstractServer, AbstractTrainer, GeneralEvaluator
from entity.server import GeneralServer, CKKSServer
from entity.client import GeneralClient, CKKSClient
from model.net import MLP, CNNMnist, CNNCifar
from split.dataset import mnist, cifar10
from split.sampling import mnist_balance_iid, cifar10_balance_iid

import numpy as np
import tenseal as ts

from util.option import args_parser
from torchsummary import summary


def print_info(flag: int):
    if flag == 1:
        print('==========================1.初始化参数==========================')
    elif flag == 2:
        print('==========================2.生成数据集==========================')
    elif flag == 3:
        print('==========================3.初始化网络==========================')
    elif flag == 4:
        print('==========================4.生成辅助上下文=======================')
    elif flag == 5:
        print('==========================5.初始化服务器========================')
    elif flag == 6:
        print('==========================6.划分数据集==========================')
    elif flag == 7:
        print('==========================7.初始化客户端========================')
    elif flag == 8:
        print('==========================8.初始化评估器========================')
    elif flag == 9:
        print('==============================================================')
        print('===========================仿真正式开始=========================')
        print('==============================================================')
        print()


class Simulator(object):
    """
    仿真器，进行Fed整体过程的仿真
    """

    def __init__(self, args):
        print_info(1)
        self.args = args
        # 1.记录仿真客户端数
        self.num_clients = args.num_clients
        print_info(2)
        # 2.选择数据集
        self.dataset_train, self.dataset_test, self.input_dim, self.channels, self.num_labels = \
            self.init_dataset()
        # 3.生成训练网络
        print_info(3)
        self.net = self.init_net()
        # 4.生成加密上下文，如果采样密态聚合就返回ckks_context,如果采样明文聚合就返回None
        print_info(4)
        self.ckks_context = self.init_ckks_context()
        print_info(5)
        # 5.根据聚合策略生成服务器
        self.agg_server = self.choice_agg_server()
        print_info(6)
        # 6.对数据集进行划分
        self.partition_items = self.partition_data()
        print_info(7)
        # 7.生成客户端列表
        self.client_list = self.init_agg_clients()
        print_info(8)
        # 8.构建模型评估器
        self.evaluator = self.init_evaluator()
        print_info(9)

    def init_dataset(self):
        """
        选择数据集
        :return: Tuple(训练集、测试集、输入大小、输入通道、输出大小)
        """
        if self.args.dataset == 'mnist':
            return mnist()
        if self.args.dataset == 'cifar10':
            return cifar10()
        raise ValueError('dataset ', self.args.dataset, 'is not supported')

    def init_net(self):
        """
        选择神经网络
        :return:
        """
        net = None
        if self.args.model == 'mlp':
            net = MLP(self.input_dim * self.channels, self.num_labels)
            summary(net, input_size=(self.channels, self.input_dim))
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            net = CNNMnist()
            summary(net, input_size=(1, 28, 28))
        elif self.args.model == 'cnn' and self.args.dataset == 'cifar10':
            net = CNNCifar()
            summary(net, input_size=(3, 32, 32))
        else:
            raise ValueError('model ', self.args.model, 'is not supported')
        return net

    def choice_agg_server(self) -> AbstractServer:
        if self.args.strategy == 'ckks_sided_discard':
            return CKKSServer(self.args, self.net.state_dict(), self.ckks_context)
        return GeneralServer(self.args, self.net.state_dict())

    def partition_data(self):
        if self.args.dataset == 'mnist':
            if self.args.partition == 'iid':
                return mnist_balance_iid(self.dataset_train, self.num_clients)
            if self.args.partition == 'dir':
                return None

        if self.args.partition == 'cifar10':
            if self.args.partition == 'iid':
                return cifar10_balance_iid(self.dataset_train, self.num_clients)
            if self.args.partition == 'dir':
                return None

    def init_agg_clients(self) -> List[AbstractTrainer]:
        nums = self.num_clients
        clients_list = []
        malicious_upper = int(self.args.malicious_frac * nums)
        malicious_clients = np.random.choice(range(nums), malicious_upper, replace=False)
        for i in range(nums):
            malicious = i in malicious_clients
            if self.args.strategy == 'ckks_sided_discard':
                client = CKKSClient(i, self.dataset_train, self.partition_items[i], self.num_labels, self.args,
                                    self.net, malicious, self.ckks_context)
            else:
                client = GeneralClient(i, self.dataset_train, self.partition_items[i], self.num_labels, self.args,
                                       self.net, malicious)
            clients_list.append(client)
        return clients_list

    def init_ckks_context(self):
        if self.args.strategy == 'ckks_sided_discard':
            bits_scale = 20
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=4096,
                coeff_mod_bit_sizes=[24, bits_scale, bits_scale, bits_scale, 24]
            )
            context.global_scale = 2 ** bits_scale
            context.generate_galois_keys()
            return context
        return None

    def init_evaluator(self):
        return GeneralEvaluator(self.net, self.dataset_test, self.args.eval_batch_size, self.args.device)

    def start(self):
        epochs = self.args.epochs
        server = self.agg_server
        client_list = self.client_list
        server.get_global_parameters()
        evaluator = self.evaluator
        for epoch in range(epochs):
            local_update_list = []
            local_loss_list = []

            print('=====================epoch {:d} start========================='.format(epoch))
            # 1. 获取全局参数
            global_parameters = server.get_global_parameters()
            # 2. 选择客户端
            selected_client_list = server.select_clients()
            print('select_client_list : ', [index for index in selected_client_list])
            # 3. 客户端本地训练
            print('client_local_update : ', end='')
            for selected_client in selected_client_list:
                client = client_list[selected_client]
                # 3.1 客户端本地更新全局参数
                client.set_parameters(global_parameters)
                # 3.2 local_update
                local_update, local_loss = client.train()
                local_update_list.append(local_update)
                local_loss_list.append(local_loss)
                print(selected_client, end='\t')
            # 4. 服务器聚合
            print('\nserver start aggregate ')
            server.aggregate(local_update_list)
            # 5.统计训练损失
            print('train avg loss {:.8f}'
                  .format(sum(local_loss_list) / len(local_loss_list)))
            # 6.参数评估
            acc = evaluator.evaluate(server.get_global_parameters())
            print('test avg acc {:.8f}-\n'
                  .format(acc))

    def check_args(self, args):
        pass


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    simulator = Simulator(args)
    simulator.start()
