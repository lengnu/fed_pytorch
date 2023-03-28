import collections
from typing import List, OrderedDict, Union

import torch
from torch import Tensor
import tenseal as ts

from aggregate.abstract import AbstractAggregator


class TrimmedMeanAggregator(AbstractAggregator):

    def __init__(self, args):
        super().__init__(args)
        self.malicious_upper = int(args.num_clients * args.malicious_frac)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        return self._trimmed(client_updates)

    def _trimmed(self, client_updates: List[OrderedDict[str, Tensor]]) -> OrderedDict[str, Tensor]:
        """
        裁剪平均聚合算法
        :param client_updates:
        :return:
        """
        num_updates = len(client_updates)
        neural_network_levels = client_updates[0].keys()
        # 2. 进行裁剪平均
        aggregate_model = collections.OrderedDict()
        for neural_level in neural_network_levels:
            cur_level_update_list = []
            for client_index in range(num_updates):
                cur_level_update_list.append(client_updates[client_index][neural_level])
            aggregate_model[neural_level] = self._trimmed_mean(cur_level_update_list, num_updates, self.malicious_upper)
        return aggregate_model

    def _trimmed_mean(self, tensors: List[Tensor], num_clients: int, malicious_upper: int) -> Tensor:
        """ 计算裁剪过后的均值 """
        trimmed_tensor, _ = torch.sort(torch.stack(tensors, dim=0), dim=0)
        return torch.mean(trimmed_tensor[malicious_upper:num_clients - malicious_upper], dim=0)
