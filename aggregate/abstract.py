"""
@FileName：abstract.py
@Description：
@Author：duwei
@Time：2023/3/28 9:49
@Email: 1456908874@qq.com
"""
from abc import ABC, abstractmethod
from typing import Union, List, OrderedDict, final

import tenseal as ts
from torch import Tensor


class AbstractAggregator(ABC):
    """
    抽象聚合器
    """

    def __init__(self, args):
        self._check_init(args)
        self.num_clients = args.num_clients
        self.client_ids = range(self.num_clients)
        self.device = args.device
        self.args = args

    @final
    def aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        """
        模型聚合（模板方法）：
            1.  首先进行健康性校验，例如恶意客户端数量不能超过聚合算法容忍上限，或者进行一些可验证聚合的Verify，校验过程由子类实现
            2.  对模型参数进行聚合，具体过程由子类实现，如果传入的是Tensor，则是明文聚合；传入CKKSTensor则是密文聚合
        """
        self.check(client_updates)
        return self._raw_aggregate(client_updates)

    @abstractmethod
    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        """
        聚合算法，由子类去实现
        """
        pass

    @final
    def check(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]):
        """
        健壮性校验，模板方法，子类不用关心
        """
        self._check_non_null(client_updates)
        self._check_active(client_updates)

    @final
    def _check_non_null(self, client_updates: List):
        """
        检查客户端更新列表是否为空
        """
        if len(client_updates) <= 0:
            raise ValueError('client_updates length cannot be less than or equal to 0')

    def _check_active(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]):
        """
        具体健壮性校验过程，可以具体的聚合子类进行覆盖（FedAvg需要覆盖）
        """
        if len(client_updates) != self.num_clients:
            raise ValueError('client_updates are not enough')

    def choice_clients(self):
        """
        选择用于下一次参与聚合的客户端
        :return:   被选中客户端id列表
        """
        return self.client_ids

    def _check_init(self, args):
        """
        初始化参数校验，子类可以进行扩展
        :param args:
        :return:
        """
        if args.num_clients <= 0:
            raise ValueError('num_clients cannot be less than or equal to 0')
