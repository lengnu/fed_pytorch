"""
@FileName：fed_avg.py
@Description：
@Author：duwei
@Time：2023/3/13 8:56
@Email: 1456908874@qq.com
"""
import copy
from typing import List, OrderedDict, Union

import numpy as np
from torch import Tensor
import tenseal as ts

from aggregate.abstract import AbstractAggregator


class FedAvgAggregator(AbstractAggregator):
    """
    联邦平均
    """

    def __init__(self, args):
        super().__init__(args)

    def choice_clients(self):
        return np.random.choice(self.client_ids, int(self.args.select_frac * self.num_clients), replace=False)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        return self._fed_avg(client_updates)

    def _check_active(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]):
        if len(client_updates) != int(self.args.select_frac * self.num_clients):
            raise ValueError('client_updates are not enough')

    def _fed_avg(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        global_parameters = copy.deepcopy(client_updates[0])
        num_clients = len(client_updates)
        for level in global_parameters.keys():
            for client in range(1, num_clients):
                global_parameters[level] += client_updates[client][level]
            global_parameters[level] /= num_clients
        return global_parameters
