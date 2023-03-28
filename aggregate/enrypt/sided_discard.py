import copy
from typing import OrderedDict, List

import torch
import tenseal as ts
from tenseal import CKKSTensor


def sided_discard(client_updates: List[OrderedDict[str, CKKSTensor]], device, malicious_upper, context):
    pass


# 1.获取CKKSTensor的均值
def mean_ckks_tensor(client_updates: List[OrderedDict[str, CKKSTensor]]):
    num_clients = len(client_updates)
    sum_ckks_tensor = copy.deepcopy(client_updates[0])
    for level in sum_ckks_tensor.keys():
        for index in range(1, num_clients):
            sum_ckks_tensor[level] += client_updates[index][level]
        sum_ckks_tensor[level] = sum_ckks_tensor[level].mul(1 / num_clients)
    return sum_ckks_tensor


# 2.计算每个客户端距离均值的距离
def distance_ckks_tensor(client_updates: List[OrderedDict[str, CKKSTensor]]
                         , mean_update: OrderedDict[str, CKKSTensor]):
    num_clients = len(client_updates)
    torch.zeros()


if __name__ == '__main__':
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    client_1 = {
        'conv1': ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float)),
        'conv2': ts.ckks_tensor(context, torch.ones(3, dtype=torch.float) * 7),
    }

    client_2 = {
        'conv1': ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float) * 3),
        'conv2': ts.ckks_tensor(context, torch.ones(3, dtype=torch.float) * 2),
    }

    client_updates = [client_1, client_2]
    for level, model in mean_ckks_tensor(client_updates, context).items():
        print('level : ')
        print(model.decrypt().tolist())
