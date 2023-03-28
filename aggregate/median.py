import collections
from typing import List, OrderedDict, Union

import torch
from torch import Tensor
import tenseal as ts

from aggregate.abstract import AbstractAggregator


class MedianAggregator(AbstractAggregator):
    def __init__(self, args):
        super().__init__(args)
        self.malicious_upper = int(args.num_clients * args.malicious_frac)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        return self._median(client_updates)

    def _median(self, client_updates: List[OrderedDict[str, Tensor]]) -> OrderedDict[str, Tensor]:
        """
        中位数聚合
        :param client_updates:
        :return:
        """
        neural_network_levels = client_updates[0].keys()
        # 2. 遍历所有客户端模型更新求取均值
        client_count = len(client_updates)
        aggregate_model = collections.OrderedDict()
        for neural_level in neural_network_levels:
            cur_level_update_list = []
            for client_index in range(client_count):
                cur_level_update_list.append(client_updates[client_index][neural_level])
            aggregate_model[neural_level] = torch.quantile(torch.stack(cur_level_update_list, dim=0), dim=0, q=0.5)
        return aggregate_model
