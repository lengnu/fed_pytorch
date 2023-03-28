import tenseal as ts
import torch

if __name__ == '__main__':
    v1 = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

    v2 = torch.tensor([
        [4, 5, 6],
        [1, 2, 3]
    ])

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    enc_v1 = ts.ckks_tensor(context, v1)
    enc_v2 = ts.ckks_tensor(context, v2)

    dec_v1 = enc_v1.decrypt()
    dec_v1_list = dec_v1.tolist()
    print(dec_v1_list)
    print(dec_v1)
