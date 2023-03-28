"""
@FileName：abstract.py
@Description：
@Author：duwei
@Time：2023/3/28 9:49
@Email: 1456908874@qq.com
"""
import collections
import copy
from abc import ABC, abstractmethod
from typing import Union, List, OrderedDict

import torch
import tenseal as ts
from torch import Tensor
import numpy as np

import util.option


class Aggregator(ABC):
    """
    抽象聚合器
    """

    def __init__(self, args):
        if args.num_clients <= 0:
            raise ValueError('num_clients cannot be less than or equal to 0')
        self.num_clients = args.num_clients
        self.client_ids = range(self.num_clients)
        self.device = args.device
        self.args = args

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

    def check(self, client_updates: List[OrderedDict[str, Union[Tensor, ts.CKKSTensor]]]):
        """
        健壮性校验，模板方法，子类不用关心
        """
        self._check_non_null(client_updates)
        self._check_active(client_updates)

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


class FedAvgAggregator(Aggregator):
    """
    联邦平均
    """

    def __init__(self, args):
        super().__init__(args)

    def choice_clients(self):
        return np.random.choice(int(self.args.select_frac * self.num_clients), self.client_ids, replace=False)

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


class KrumAggregator(Aggregator):
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


class SidedDiscardAggregator(Aggregator):
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
        for level in global_parameters.keys():
            for id in selected_client_ids:
                if id != selected_client_ids[0]:
                    global_parameters[level] += client_updates[id][level]
            global_parameters[level] /= len(selected_client_ids)
        return global_parameters

    def _update_mean(self, client_updates: List[OrderedDict[str, Tensor]]) -> OrderedDict[str, Tensor]:
        """
        计算客户端更新的均值
        """
        client_update_mean = copy.deepcopy(client_updates[0])
        num_updates = len(client_updates)
        for level in client_update_mean.keys():
            for client in range(1, num_updates):
                client_update_mean[level] += client_updates[client][level]
            client_update_mean[level] /= num_updates
        return client_update_mean

    def _distance_mean(self, client_updates: List[OrderedDict[str, Tensor]],
                       update_mean: OrderedDict[str, Tensor]) -> Tensor:
        """
        计算每个客户端更新距离均值的距离
        """
        num_updates = len(client_updates)
        distance_list = torch.zeros(num_updates, dtype=torch.float).to(self.device)
        for level in update_mean.keys():
            for client_id in range(num_updates):
                distance_list[client_id] += euclidean_distance(client_updates[client_id][level], update_mean[level])
        return distance_list

    def _select_clients(self, distance_list: Tensor) -> List[int]:
        _, indexes = torch.sort(distance_list)
        indexes = indexes[self.malicious_upper:self.num_clients - self.malicious_upper]
        selected_client_list = indexes.detach().numpy().tolist()
        return selected_client_list


class SidedDiscardEncryptAggregator(Aggregator):

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
        ckks_num_clients = ts.ckks_tensor(self.context,torch.tensor([1.0  / len(client_updates)],dtype=torch.float).to(self.device))
        for level in global_parameters.keys():
            for id in selected_client_ids:
                if id != selected_client_ids[0]:
                    global_parameters[level] += client_updates[id][level]
            global_parameters[level] *= ckks_num_clients
        return global_parameters

    def _ckks_update_mean(self, client_updates: List[OrderedDict[str, ts.CKKSTensor]]) -> \
            OrderedDict[str, ts.CKKSTensor]:
        client_update_mean = copy.deepcopy(client_updates[0])
        num_updates = len(client_updates)
        for level in client_update_mean.keys():
            for client in range(1, num_updates):
                client_update_mean[level] += client_updates[client][level]
            client_update_mean[level] *= (1.0 / num_updates)
        return client_update_mean

    def _ckks_distance_mean(self, client_updates: List[OrderedDict[str, ts.CKKSTensor]],
                            ckks_update_mean: OrderedDict[str, ts.CKKSTensor]) -> List[ts.CKKSTensor]:
        num_updates = len(client_updates)
        distance_list = [ts.ckks_tensor(self.context, torch.tensor([0], dtype=torch.float).to(self.device))
                         for _ in range(num_updates)]
        for level in ckks_update_mean.keys():
            for client_id in range(num_updates):
                distance_list[client_id] += (client_updates[client_id][level] - ckks_update_mean[level]).pow(2).sum()
        return distance_list

    def _ckks_selects_clients(self, ckks_distance_list: List[ts.CKKSTensor]) -> List[int]:
        distance_plain_list = torch.tensor([ckks_dis.decrypt().tolist()[0] for ckks_dis in ckks_distance_list],
                                           dtype=torch.float).to(self.device)

        _, indexes = torch.sort(distance_plain_list)
        indexes = indexes[self.malicious_upper:self.num_clients - self.malicious_upper]
        selected_client_list = indexes.detach().numpy().tolist()
        return selected_client_list


def euclidean_distance(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    计算两个tensor之间欧式距离
    :param tensor_1:   tensor1
    :param tensor_2:   tensor2
    :return:    欧氏距离
    """
    return (tensor_1 - tensor_2).pow(2).sum()


if __name__ == '__main__':
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    # ts_1 = ts.ckks_tensor(context, torch.tensor([1, 2, 3, 4], dtype=torch.float))
    # ts_2 = ts.ckks_tensor(context, torch.tensor([1, 2, 3, 4], dtype=torch.float))
    # ts_1 += ts_2
    # print(ts_1.decrypt().tolist())

    clients_updates = []
    client_1_update = collections.OrderedDict()
    client_1_update['1'] = torch.zeros(2, 2, dtype=torch.float)
    client_1_update['2'] = torch.ones(2, 2, dtype=torch.float)
    client_1_update['3'] = torch.tensor([
        [[1, 2], [3, 4]],
        [[9, 7], [7.56, 2.4]],
    ], dtype=torch.float)

    client_2_update = collections.OrderedDict()
    client_2_update['1'] = torch.ones(2, 2, dtype=torch.float)
    client_2_update['2'] = torch.ones(2, 2, dtype=torch.float)
    client_2_update['3'] = torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 4], [2, 6]],
    ], dtype=torch.float)

    clients_updates.append(client_1_update)
    clients_updates.append(client_2_update)
    args = util.option.args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    client__updates_ckks = []
    client_1_update_ckks = collections.OrderedDict()
    client_1_update_ckks['1'] = ts.ckks_tensor(context, torch.zeros(2, 2, dtype=torch.float).flatten())
    print(client_1_update_ckks['1'])
    client_1_update_ckks['2'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
    client_1_update_ckks['3'] = ts.ckks_tensor(context, torch.tensor([
        [[1, 2], [3, 4]],
        [[9, 7], [7.56, 2.4]],
    ], dtype=torch.float).flatten())

    client_2_update_ckks = collections.OrderedDict()
    client_2_update_ckks['1'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
    client_2_update_ckks['2'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
    client_2_update_ckks['3'] = ts.ckks_tensor(context, torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 4], [2, 6]],
    ], dtype=torch.float).flatten())

    client__updates_ckks.append(client_1_update_ckks)
    client__updates_ckks.append(client_2_update_ckks)


    aggregator_1 = SidedDiscardAggregator(args)
    aggregator_2 = SidedDiscardEncryptAggregator(context, args)
    print(aggregator_1.aggregate(clients_updates))
    print('\n\n')
    agg_ckks = aggregator_2.aggregate(client__updates_ckks)

    for level,model in agg_ckks.items():
        print('level = ',level,'\t\t model = ' ,model.decrypt().tolist())
