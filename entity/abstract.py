import collections
import copy
import time
from abc import ABC, abstractmethod
from typing import OrderedDict, List, Union, final

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import tenseal as ts

from aggregate.abstract import AbstractAggregator
from split.splitter import DatasetSplitter


class AbstractServer(ABC):
    """
    一个抽象的服务器，进行聚合
    """

    def __init__(self, args, init_parameters, context=None):
        self.check_init(args, init_parameters)
        self.args = args
        self.context = context
        self.global_parameters = self.init_params(init_parameters)
        self.aggregator = self._choice_aggregator(args)

    def init_params(self, init_parameters):
        return copy.deepcopy(init_parameters)

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

    @final
    def _choice_aggregator(self, args) -> AbstractAggregator:
        """
        导入服务端的聚合器
        :param args:    聚合策略
        :return:
        """
        # 由具体实现类去进行选择
        return self._find_aggregator(args)

    @abstractmethod
    def _find_aggregator(self, args) -> AbstractAggregator:
        """
        子类重写该方法以便返回一个聚合器
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

    @final
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
    """
    抽象的训练器，由不同的Client继继承实例化
    """

    def __init__(self,
                 client_id,
                 dataset,
                 partition_items,
                 num_labels,
                 args,
                 net,
                 malicious=False,
                 context=None
                 ):
        """
        :param client_id:       client标识，全局唯一
        :param dataset:         数据集 MNIST/CIFAR10/...
        :param partition_items: 客户端的数据集元素索引,dataset加载时只取partition_items中的图片
        :param num_labels:      数据集类别数量
        :param args:            仿真参数
        :param net:             神经网络
        :param malicious:       客户端是否是恶意的，只有malicious为True参数中配置的攻击才生效
        """
        self.id = client_id
        self.num_labels = num_labels
        self.args = args
        self.net = copy.deepcopy(net)
        self.malicious = malicious
        self.net_meta = self.get_net_meta()
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.data_loader = DataLoader(DatasetSplitter(dataset, partition_items, args, malicious),
                                      batch_size=args.local_batch_size, shuffle=True)
        self.context = context
        self._after_properties()

    def _after_properties(self):
        """
        留给子类做一些额外工作，例如修改父类定义的损失函数，加载器等
        """
        pass

    def get_net_meta(self) -> OrderedDict[str, Tensor]:
        """
        获取神经网络每一层的tensor结构
        :return:    神经网络结果，包括层名称和大小
        """
        net_meta = collections.OrderedDict()
        for neural_level, model in self.net.state_dict().items():
            net_meta[neural_level] = model.shape
        return net_meta

    def update_parameters(self, global_parameters) -> None:
        """
        根据全局模型更新本地网络
        :param global_parameters:   全局更新
        :return:
        """
        parameters = copy.deepcopy(global_parameters)
        self.net.load_state_dict(parameters)

    def get_parameters(self):
        local_update = collections.OrderedDict()
        if self.malicious:
            if self.args.inert_enable:
                # 如果开启了惰性攻击，则随机生成网络参数
                gauss_mean = self.args.gauss_mean
                gauss_std = self.args.gauss_std
                for neural_level, shape in self.net_meta.items():
                    local_update[neural_level] = torch.normal(mean=gauss_mean, std=gauss_std, size=shape).to(
                        self.args.device)
            elif self.args.gradient_scale_enable:
                # 梯度缩放攻击
                scale_factor = torch.tensor(self.args.scale, dtype=torch.float).to(self.args.device)
                local_update = self.net.state_dict() * scale_factor
        else:
            # 无攻击，直接提取网络参数
            local_update = self.net.state_dict()
        return local_update

    @final
    def get_global_parameters(self):
        return self.net.state_dict()

    def train(self):
        # start = time.time()
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
        # print('train time ',time.time() - start)
        return self.get_parameters(), sum(epoch_loss) / len(epoch_loss)


class AbstractEvaluator(ABC):
    """
    评估测试集
    """

    def __init__(self, net, dataset_test, batch_size, device):
        self.net = copy.deepcopy(net)
        self.dataset_test = dataset_test
        self.data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        self.device = device
        self.loss_func = self._init_loss_func()

    def _init_loss_func(self):
        """
        设置损失函数，如果需要不同的损失函数由子类实现
        :return:    损失函数
        """
        return nn.CrossEntropyLoss().to(self.device)

    def evaluate(self, parameters):
        """
        模型评估
        :param parameters: 评估参数
        :return:    模型精度
        """
        self.net.load_state_dict(copy.deepcopy(parameters))
        return self._evaluate_acc()

    def _evaluate_acc(self):
        """
        评估模型
        :return:    模型精度
        """
        net = self.net
        net.eval()
        accurate_count = 0.0
        total_count = len(self.data_loader.dataset)
        for batch, (items, labels) in enumerate(self.data_loader):
            items, labels = items.to(self.device), labels.to(self.device)
            predictive_labels = net(items)
            accurate_count += (predictive_labels.argmax(dim=1) == labels).sum().item()
        return accurate_count / total_count


class GeneralEvaluator(AbstractEvaluator):
    def __init__(self, net, dataset_test, batch_size, device):
        super().__init__(net, dataset_test, batch_size, device)
