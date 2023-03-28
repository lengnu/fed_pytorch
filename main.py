import numpy as np
import torch

from entity.client import SyncClient
from entity.server import SyncServer
from model.net import CNNCifar, CNNMnist
from split.dataset import mnist, cifar10
from split.sampling import cifar10_balance_iid
from util.option import args_parser
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = args_parser()
    args.device = device

    # 设置数据集和模型
    if args.dataset == 'mnist':
        dataset_train, dataset_test, num_labels = mnist()
        global_net = CNNMnist()
    else:
        dataset_train, dataset_test, num_labels = cifar10()
        global_net = CNNCifar()

    # 初始化参数
    num_clients = args.num_clients
    malicious_upper = int(args.malicious_frac * num_clients)
    malicious_client_id = np.random.choice(range(num_clients), malicious_upper, replace=False)

    # 1.初始化参数
    init_parameters = global_net.state_dict()

    # 2.分割数据集
    partition_clients = cifar10_balance_iid(dataset_train, num_clients)

    # 3.创建服务器和客户端
    server = SyncServer(args, init_parameters)
    client_list = []
    for client_id in range(num_clients):
        if client_id in malicious_client_id:
            client = SyncClient(client_id, dataset_train, partition_clients[client_id], num_labels, args, global_net,
                                True)
        else:
            client = SyncClient(client_id, dataset_train, partition_clients[client_id], num_labels, args, global_net,
                                False)
        client_list.append(client)

    # 4.开始训练
    global_train_loss = []
    global_test_loss = []
    global_test_acc = []
    for epoch in range(args.epochs):
        # 获取全局模型
        global_parameters = server.get_global_parameters()
        # 随机挑选一部分客户端
        selected_clients = server.select_clients()
        local_update_list = []
        local_loss_list = []
        for selected_client_id in selected_clients:
            client = client_list[selected_client_id]
            client.set_parameters(global_parameters)
            local_update, local_loss = client.train()
            local_update_list.append(local_update)
            local_loss_list.append(local_loss)
        # 服务器开始聚合
        server.aggregate(local_update_list)

        # 计算平均损失
        print('第 {:d} 次训练，训练平均损失为: {:.8f}'.format(epoch, sum(local_loss_list) / len(local_loss_list)))
