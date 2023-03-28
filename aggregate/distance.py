"""
@FileName：distance.py
@Description：距离计算
@Author：duwei
@Time：2023/3/6 17:24
@Email: 1456908874@qq.com
"""
from typing import OrderedDict, List

import torch
from torch import Tensor


def batch_model_flatten(models_params: List[OrderedDict[str, Tensor]]) -> List[Tensor]:
    """ 将所有神经网络的参数展开为向量 """
    models = []
    for model_params in models_params:
        models.append(model_flatten(model_params))
    return models


def model_flatten(model_params: OrderedDict[str, Tensor]) -> Tensor:
    """ 将神经网络模型参数展开为1维向量 """
    return torch.cat([torch.flatten(param) for param in model_params.values()])


def euclidean_distance_flatten_tensor(flatten_tensor_1: Tensor, flatten_tensor_2: Tensor) -> Tensor:
    return torch.pow((flatten_tensor_1 - flatten_tensor_2), 2).sum()


def norm_distance(tensor_1: Tensor, tensor_2: Tensor, p=2) -> Tensor:
    return torch.norm(tensor_1 - tensor_2, p)


def cos_dis_flatten_tensor(flatten_tensor_1: Tensor, flatten_tensor_2: Tensor) -> Tensor:
    """ 求两个展开为1维tensor余弦相似度 """
    return torch.cosine_similarity(flatten_tensor_1, flatten_tensor_2, dim=0)


def euclidean_distance(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    return torch.sum(torch.pow(tensor_1 - tensor_2, 2) )


# def tensor_european_distance(model_params_1: OrderedDict[str, Tensor],
#                              model_params_2: OrderedDict[str, Tensor]) -> Tensor:
#     """ 求两个参数更新之间欧式距离 """
#     return torch.norm(model_flatten(model_params_1) - model_flatten(model_params_2), p=2)
