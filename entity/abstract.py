import collections
import copy
from abc import ABC, abstractmethod
from typing import OrderedDict, List, Union, final

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import tenseal as ts

from aggregate.abstract import AbstractAggregator
from aggregate.fed_avg import FedAvgAggregator
from aggregate.krum import KrumAggregator
from aggregate.sided_discard import SidedDiscardAggregator, SidedDiscardEncryptAggregator
from aggregate.median import MedianAggregator
from aggregate.trimmed_mean import TrimmedMeanAggregator
from split.splitter import DatasetSplitter


class AbstractServer(ABC):
    """
    一个抽象的服务器，进行聚合
    """

    def __init__(self, args, init_parameters):
        self.check_init(args, init_parameters)
        self.args = args
        self.global_parameters = init_parameters
        self.aggregator = self._choice_aggregator(args)

    @final
    def check_init(self, args, init_parameters) -> None:
        """
        初始化服务器需要进行检查
        :param args:            聚合的仿真参数
        :param init_parameters: 初始化模型
        """
        if init_parameters is None:
            raise ValueError('init_parameters must be not None')
        self._check_args(args)

    def _check_args(self, args) -> None:
        """
        对聚合的仿真参数进行校验，子类有需要就去进行具体实现
        :param args: 聚合的仿真参数
        """
        pass

    def _choice_aggregator(self, args) -> AbstractAggregator:
        """
        导入服务端的聚合器
        :param args:    聚合策略
        :return:
        """
        # 先由子类进行寻找
        aggregator = self._find_aggregator(args)
        if aggregator is not None:
            return aggregator
        # 子类找不到父类进行加载
        if args.strategy == 'fed_avg':
            return FedAvgAggregator(args)
        elif args.strategy == 'krum':
            return KrumAggregator(args)
        elif args.strategy == 'median':
            return MedianAggregator(args)
        elif args.strategy == 'trimmed_mean':
            return TrimmedMeanAggregator(args)
        elif args.strategy == 'sided_discard':
            return SidedDiscardAggregator(args)
        elif args.strategy == 'sided_discard_enc':
            return SidedDiscardEncryptAggregator(args)
        else:
            raise ValueError('There is no such aggregation strategy ', args.strategy)

    def _find_aggregator(self, args) -> AbstractAggregator:
        """
        子类需要使用额外的聚合器时，就重写该方法去自行扩展
        :param args:    仿真参数
        :return:        聚合器
        """
        pass

    @final
    def aggregate(self, client_updates, **kwargs):
        """
        聚合整体流程
            1. 为了方便拓展，例如进行验证、噪声消除等功能，提供一个before_aggregate方法
            2. 服务器进行聚合得到全局更新，并将全局更新设置到自己的参数中
            3. 聚合后有可能做一些其他的善后工作，提供一个post_aggregate方法
            4. before_aggregate方法和post_aggregate方法两个方法都由子类实现，目前的聚合算法不需要这两个方法
        :param client_updates: 客户端更新列表
        :param kwargs:  其他额外的辅助参数
        :return:
        """
        self._before_aggregate(client_updates, **kwargs)
        global_parameters = self._aggregate_update(client_updates)
        self._set_global_parameters(global_parameters)
        self._post_aggregate(client_updates, **kwargs)

    @final
    def _aggregate_update(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        """
        对客户端更新进行聚合
        :param client_updates:  客户端更新列表
        :return: 聚合后的全局更新
        """
        return self.aggregator.aggregate(client_updates)

    def select_clients(self):
        """
        选择下一轮训练客户端，由具体的聚合算法进行选择
        :return:
        """
        return self.aggregator.choice_clients()

    @final
    def get_global_parameters(self):
        """
        获取当前全局参数
        :return:    全局参数（模型或者梯度）
        """
        return self.global_parameters

    def _set_global_parameters(self, global_parameters) -> None:
        """
        将聚合结果设置位全局参数，该方法在模型聚合时不要考虑，主要用于服务的聚合梯度时需要更新全局模型
        如果有需要可由子类覆盖
        :param global_parameters: 聚合后的全局更新（模型或者参数）
        """
        self.global_parameters = global_parameters

    @abstractmethod
    def _before_aggregate(self, client_updates, **kwargs):
        pass

    @abstractmethod
    def _post_aggregate(self, client_updates, **kwargs):
        pass


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
