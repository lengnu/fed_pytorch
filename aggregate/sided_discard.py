"""
@FileName：sided_discard.py
@Description：双端丢弃的聚合方式
@Author：duwei
@Time：2023/3/8 15:22
@Email: 1456908874@qq.com
"""
import collections
from typing import List, OrderedDict

import torch
from torch import Tensor

from aggregate.distance import batch_model_flatten, norm_distance, euclidean_distance


def sided_discard(client_updates: List[OrderedDict[str, Tensor]], malicious_upper=0) -> OrderedDict[str, Tensor]:
    """ 双端丢弃的聚合方案 """
    client_count = len(client_updates)
    # 1.展开模型
    client_flatten_updates = batch_model_flatten(client_updates)
    # 2.计算模型均值
    client_mean_update = torch.mean(torch.stack(client_flatten_updates, dim=0), dim=0)
    # 3.计算每个模型距离均值的距离
    distance_tensor = _distance_mean(client_flatten_updates, client_mean_update)
    # 4.计算参与聚合的客户端列表
    selected_client_list = _model_weights(distance_tensor, client_count, malicious_upper=malicious_upper)
    aggregate_model = collections.OrderedDict()
    neural_network_levels = client_updates[0].keys()
    for level in neural_network_levels:
        cur_level_model = torch.zeros_like(client_updates[0][level])
        for selected_client in selected_client_list:
            cur_level_model += client_updates[selected_client][level]
        aggregate_model[level] = torch.div(cur_level_model, len(selected_client_list))
    return aggregate_model


def _distance_mean(client_update_list: List[Tensor], mean_update: Tensor) -> Tensor:
    client_count = len(client_update_list)
    distance_list = []
    for cur_client_index in range(client_count):
        distance_list.append(
            euclidean_distance(client_update_list[cur_client_index], mean_update)
        )
    return torch.stack(distance_list, dim=0)


def _model_weights(distance_list: Tensor, client_count: int, malicious_upper: int) -> List[int]:
    """计算每个客户端的权重系数"""
    sorted_dis, indexes = torch.sort(distance_list)
    indexes = indexes[malicious_upper:client_count - malicious_upper]
    selected_client_list = indexes.detach().numpy().tolist()
    return selected_client_list
