import copy
from typing import List, OrderedDict, Union

import torch
from torch import Tensor
import tenseal as ts

from aggregate.abstract import AbstractAggregator
from util.compute import euclidean_distance


class KrumAggregator(AbstractAggregator):
    """
    Krum聚合算法
    """

    def __init__(self, args):
        super().__init__(args)
        self.malicious_upper = int(args.num_clients * args.malicious_frac)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        return self._krum(client_updates)

    def _krum(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]):
        """
        Krum聚合算法
        :param client_updates:  客户端更新
        :return:    聚合后模型
        """
        num_updates = len(client_updates)
        # 1.计算客户端之间两两距离
        distance_map = self._create_distance(client_updates, num_updates)
        # 2.计算每个客户端的得分\
        scores = self._calculate_score(distance_map, num_updates)
        # 3.选择得分最小的客户端更新作为全局更新
        selected_client = torch.argmin(scores).item()
        return copy.deepcopy(client_updates[selected_client])

    def _create_distance(self, client_updates: List[OrderedDict[str, Tensor]], num_updates: int) -> Tensor:
        """
        计算客户端更新之间的两两距离
        :param num_updates:     客户端更新数量
        :return:               更新之间的两两距离
        """
        distance_map = torch.zeros((num_updates, num_updates), dtype=torch.float).to(self.device)
        # 遍历每一层计算更新之间两两距离
        for level in client_updates[0].keys():
            for from_client in range(num_updates):
                for to_client in range(from_client + 1, num_updates):
                    distance = euclidean_distance(client_updates[from_client][level], client_updates[to_client][level])
                    distance_map[from_client][to_client] = distance
                    distance_map[to_client][from_client] = distance
        return distance_map

    def _calculate_score(self, distance_map: Tensor, num_updates: int) -> Tensor:
        """
        计算每个客户端的得分
        :param distance_map:    距离map
        :param num_updates:     客户端更新总数
        :return: 每个客户端的得分
        """
        sorted_distance_map, _ = torch.sort(distance_map, dim=1)
        scores = torch.sum(sorted_distance_map[:, 1:(num_updates - self.malicious_upper)], dim=1)
        return scores
