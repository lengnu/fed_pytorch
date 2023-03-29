import numpy as np
import torch

import ssl
import tenseal as ts

if __name__ == '__main__':
    tensor = torch.ones(2, 3, device=torch.device('cpu'), dtype=torch.float)
    bits_scale = 20
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[24, bits_scale, bits_scale, bits_scale, 24]
    )
    context.global_scale = 2 ** bits_scale
    context.generate_galois_keys()
    ts.ckks_tensor(context, tensor)
