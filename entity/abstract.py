import collections
import copy
from abc import ABC, abstractmethod
from typing import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from aggregate.strategy import aggregators
from split.splitter import DatasetSplitter


class AbstractAggregator(ABC):

    def __init__(self, args, init_parameters):
        self.args = args
        self.epoch_select_num = int(max(1, args.num_clients * args.select_frac))
        self.user_list = range(args.num_clients)
        if args.strategy not in aggregators.keys():
            raise ValueError('{} strategy is not exists'.format(args.strategy))
        self.aggregator = aggregators[args.strategy]
        self.malicious_upper = int(args.num_clients * args.malicious_frac)
        self.global_parameters = init_parameters

    @abstractmethod
    def aggregate(self, client_updates):
        pass

    def select_clients(self):
        return np.random.choice(self.user_list, self.epoch_select_num, replace=False)

    def get_global_parameters(self):
        return self.global_parameters


class AbstractTrainer(ABC):
    def __init__(self,
                 client_id,
                 dataset,
                 partition,
                 num_labels,
                 args,
                 net,
                 malicious=False
                 ):
        self.id = client_id
        self.num_labels = num_labels
        self.args = args
        self.net = copy.deepcopy(net)
        self.malicious = malicious
        self.net_meta = self.get_net_meta()
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.data_loader = DataLoader(DatasetSplitter(dataset, partition, args, malicious),
                                      batch_size=args.local_batch_size, shuffle=True)

    def get_net_meta(self) -> OrderedDict[str, Tensor]:
        """
        获取神经网络每一层的tensor结构
        :return:
        """
        net_meta = collections.OrderedDict()
        for level, model in self.net.state_dict().items():
            net_meta[level] = model.shape
        return net_meta

    def set_parameters(self, global_parameters):
        parameters = copy.deepcopy(global_parameters)
        self.net.load_state_dict(parameters)

    def get_parameters(self):
        local_update = collections.OrderedDict()
        if self.malicious:
            if self.args.inert_enable:
                # 如果开启了惰性攻击，则随机生成网络参数
                gauss_mean = self.args.gauss_mean
                gauss_std = self.args.gauss_std
                for level, shape in self.net_meta.items():
                    local_update[level] = torch.normal(mean=gauss_mean, std=gauss_std, size=shape)
            elif self.args.gradient_scale_enable:
                # 梯度缩放
                scale_factor = torch.tensor(self.args.scale, dtype=torch.float)
                local_update = self.net.state_dict() * scale_factor
        else:
            local_update = self.net.state_dict()
        return local_update

    def train(self):
        net = self.net
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for epoch in range(self.args.local_epochs):
            batch_loss = []
            for batch, (items, labels) in enumerate(self.data_loader):
                items, labels = items.to(self.args.device), labels.to(self.args.device)
                # 每次迭代梯度清零
                net.zero_grad()
                predictive_labels = net(items)
                loss = self.loss_func(predictive_labels, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            # 统计一次迭代的总体损失
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # 返回参数和损失大小
        return self.get_parameters(), sum(epoch_loss) / len(epoch_loss)
