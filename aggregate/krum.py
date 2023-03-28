# """
# @FileName：krum.py
# @Description：Krum拜占庭鲁棒式聚合算法
# @Author：duwei
# @Time：2023/3/6 17:23
# @Email: 1456908874@qq.com
# """
# import copy
# from typing import List, OrderedDict
#
# import torch
# from torch import Tensor
#
# from aggregate.distance import euclidean_distance
#
#
# def krum(client_updates: List[OrderedDict[str, Tensor]], device: torch.device, malicious_upper=0) -> OrderedDict[
#     str, Tensor]:
#     """
#         Krum聚合算法
#     :param device:              在CPU或者GPU上进行运算
#     :param client_updates:      所有客户端更新列表
#     :param malicious_upper:     恶意客户端上限
#     :return: 聚合后的全局更新
#     """
#     client_count = len(client_updates)
#     # 计算客户端更新之间两两聚合
#     distance_map = _create_distance(client_updates, device, client_count)
#     # 计算每个客户端得分
#     scores = _calculate_score(distance_map, client_count, malicious_upper)
#     # 选择得分最小的作为全局梯度
#     selected_client = torch.argmin(scores).item()
#     return copy.deepcopy(client_updates[selected_client])
#
#
# def _create_distance(client_updates: List[OrderedDict[str, Tensor]], device: torch.device, client_count: int) -> Tensor:
#     """
#     计算每个客户端距离其他客户端的距离
#     :return:    客户端更新之间的两两距离
#     """
#     distance_map = torch.zeros((client_count, client_count), dtype=torch.float).to(device)
#     # 遍历每一层计算更新之间两两距离
#     for level in client_updates[0].keys():
#         for from_client in range(client_count):
#             for to_client in range(from_client + 1, client_count):
#                 distance = euclidean_distance(client_updates[from_client][level], client_updates[to_client][level])
#                 distance_map[from_client][to_client] += distance
#                 distance_map[to_client][from_client] += distance
#     return distance_map
#
#
# def _calculate_score(distance_map: Tensor, client_count: int, malicious_upper: int) -> Tensor:
#     """
#     计算每个客户端的得分
#     :param distance_map:    距离map
#     :param client_count:    客户端总数
#     :param malicious_upper:  恶意客户端上限
#     :return: 每个客户端的得分
#     """
#     sorted_distance_map, _ = torch.sort(distance_map, dim=1)
#     scores = torch.sum(sorted_distance_map[:, 1:(client_count - malicious_upper)], dim=1)
#     return scores
