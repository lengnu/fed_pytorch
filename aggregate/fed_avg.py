"""
@FileName：fed_avg.py
@Description：
@Author：duwei
@Time：2023/3/13 8:56
@Email: 1456908874@qq.com
"""
import copy
from typing import List, OrderedDict

import torch
from torch import Tensor


def fed_avg(client_updates: List[OrderedDict[str, Tensor]], device, malicious_upper=0) -> OrderedDict[str, Tensor]:
    """
    联邦平均聚合
    :param client_updates:  客户端更新列表
    :param client_malicious_count:  恶意客户端上限
    :return:    聚合梯度
    """
    w_global = copy.deepcopy(client_updates[0])
    client_count = len(client_updates)
    for key in w_global.keys():
        for client in range(1, client_count):
            w_global[key] += client_updates[client][key]
        w_global[key] = torch.div(w_global[key], client_count)
    return w_global
