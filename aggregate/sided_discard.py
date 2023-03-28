"""
@FileName：sided_discard.py
@Description：双端丢弃的聚合方式
@Author：duwei
@Time：2023/3/8 15:22
@Email: 1456908874@qq.com
"""
import copy
from typing import List, OrderedDict, Union

import torch
from torch import Tensor

import tenseal as ts
from aggregate.abstract import AbstractAggregator
from util.compute import euclidean_distance


class SidedDiscardAggregator(AbstractAggregator):
    """
    双端丢弃
    """

    def __init__(self, args):
        super().__init__(args)
        self.malicious_upper = int(args.num_clients * args.malicious_frac)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]) -> \
            OrderedDict[str, Union[Tensor, ts.CKKSTensor]]:
        return self._sided_discard_aggregate(client_updates)

    def _sided_discard_aggregate(self, client_updates: List[OrderedDict[str, Tensor]]) -> OrderedDict[str, Tensor,]:
        # 1.计算更新的均值
        update_mean = self._update_mean(client_updates)
        # 2.计算每个客户端更新距离均值的距离
        distance_list = self._distance_mean(client_updates, update_mean)
        # 3.很具距离去掉最近和最远的几个客户端
        selected_client_ids = self._select_clients(distance_list)
        # 4.将选中的客户端进行平均
        global_parameters = copy.deepcopy(client_updates[selected_client_ids[0]])
        for neural_level in global_parameters.keys():
            for id in selected_client_ids:
                if id != selected_client_ids[0]:
                    global_parameters[neural_level] += client_updates[id][neural_level]
            global_parameters[neural_level] /= len(selected_client_ids)
        return global_parameters

    def _update_mean(self, client_updates: List[OrderedDict[str, Tensor]]) -> OrderedDict[str, Tensor]:
        """
        计算客户端更新的均值
        """
        client_update_mean = copy.deepcopy(client_updates[0])
        num_updates = len(client_updates)
        for neural_level in client_update_mean.keys():
            for client in range(1, num_updates):
                client_update_mean[neural_level] += client_updates[client][neural_level]
            client_update_mean[neural_level] /= num_updates
        return client_update_mean

    def _distance_mean(self, client_updates: List[OrderedDict[str, Tensor]],
                       update_mean: OrderedDict[str, Tensor]) -> Tensor:
        """
        计算每个客户端更新距离均值的距离
        """
        num_updates = len(client_updates)
        distance_list = torch.zeros(num_updates, dtype=torch.float).to(self.device)
        for neural_level in update_mean.keys():
            for client_id in range(num_updates):
                distance_list[client_id] += euclidean_distance(client_updates[client_id][neural_level],
                                                               update_mean[neural_level])
        return distance_list

    def _select_clients(self, distance_list: Tensor) -> List[int]:
        _, indexes = torch.sort(distance_list)
        indexes = indexes[self.malicious_upper:self.num_clients - self.malicious_upper]
        selected_client_list = indexes.detach().numpy().tolist()
        return selected_client_list

    def _check_init(self, args):
        super()._check_init(args)
        if args.malicious_frac >= 1 / 3:
            raise ValueError('malicious_frac should not more than 0.33')


class SidedDiscardEncryptAggregator(AbstractAggregator):

    def __init__(self, context, args):
        super().__init__(args)
        self.context = context
        self.malicious_upper = int(args.num_clients * args.malicious_frac)

    def _raw_aggregate(self, client_updates: List[OrderedDict[str, ts.CKKSTensor]]) -> \
            OrderedDict[str, ts.CKKSTensor]:
        # 1.求所有客户端更新的均值
        ckks_update_mean = self._ckks_update_mean(client_updates)
        # 2. 求每个客户端距离均值的距离
        ckks_distance_list = self._ckks_distance_mean(client_updates, ckks_update_mean)
        # 3. 去掉两端距离的更新
        selected_client_ids = self._ckks_selects_clients(ckks_distance_list)
        # 4. 对剩余客户端进行聚合
        global_parameters = copy.deepcopy(client_updates[selected_client_ids[0]])
        ckks_num_clients = ts.ckks_tensor(self.context,
                                          torch.tensor([1.0 / len(selected_client_ids)], dtype=torch.float).to(
                                              self.device))
        for neural_level in global_parameters.keys():
            for id in selected_client_ids:
                if id != selected_client_ids[0]:
                    global_parameters[neural_level] += client_updates[id][neural_level]
            global_parameters[neural_level] *= ckks_num_clients
        return global_parameters

    def _ckks_update_mean(self, client_updates: List[OrderedDict[str, ts.CKKSTensor]]) -> \
            OrderedDict[str, ts.CKKSTensor]:
        client_update_mean = copy.deepcopy(client_updates[0])
        num_updates = len(client_updates)
        for neural_level in client_update_mean.keys():
            for client in range(1, num_updates):
                client_update_mean[neural_level] += client_updates[client][neural_level]
            client_update_mean[neural_level] *= (1.0 / num_updates)
        return client_update_mean

    def _ckks_distance_mean(self, client_updates: List[OrderedDict[str, ts.CKKSTensor]],
                            ckks_update_mean: OrderedDict[str, ts.CKKSTensor]) -> List[ts.CKKSTensor]:
        num_updates = len(client_updates)
        distance_list = [ts.ckks_tensor(self.context, torch.tensor([0], dtype=torch.float).to(self.device))
                         for _ in range(num_updates)]
        for neural_level in ckks_update_mean.keys():
            for client_id in range(num_updates):
                distance_list[client_id] += (
                        client_updates[client_id][neural_level] - ckks_update_mean[neural_level]).pow(2).sum()
        return distance_list

    def _ckks_selects_clients(self, ckks_distance_list: List[ts.CKKSTensor]) -> List[int]:
        distance_plain_list = torch.tensor([ckks_dis.decrypt().tolist()[0] for ckks_dis in ckks_distance_list],
                                           dtype=torch.float).to(self.device)

        _, indexes = torch.sort(distance_plain_list)
        indexes = indexes[self.malicious_upper:self.num_clients - self.malicious_upper]
        selected_client_list = indexes.detach().numpy().tolist()
        return selected_client_list

    def _check_init(self, args):
        super()._check_init(args)
        if args.malicious_frac >= 1 / 3:
            raise ValueError('malicious_frac should not more than 1 / 3')
