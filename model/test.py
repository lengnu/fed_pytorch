"""
@FileName：test.py
@Description：
@Author：duwei
@Time：2023/3/29 16:02
@Email: 1456908874@qq.com
"""
import torch
import tenseal as ts
from torchsummary import summary

from model.net import CNNMnist, CNNCifar, MLP

if __name__ == '__main__':
    net = CNNMnist()
    summary(net, input_size=(1, 28, 28), batch_size=32)
    print()
    net = CNNCifar()
    summary(net, input_size=(3, 32, 32), batch_size=32)
    print()
    net = MLP(1 * 28 * 28, 10)
    summary(net, input_size=(1, 28, 28), batch_size=32)
    print()
    net = MLP(3 * 32 * 32, 10)
    summary(net, input_size=(3, 32, 32), batch_size=32)
    # context = ts.context(
    #     ts.SCHEME_TYPE.CKKS,
    #     poly_modulus_degree=4096,
    #     coeff_mod_bit_sizes=[24, 20, 20, 24]
    # )
    # context.generate_galois_keys()
    # context.global_scale = 2 ** 20
    # tensor = torch.randn(4097184, 1, dtype=torch.float)
    #
    # enc_1 = ts.ckks_tensor(context, tensor)
