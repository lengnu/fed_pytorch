import copy

import torch

from entity.abstract import AbstractTrainer
import tenseal as ts


class SyncClient(AbstractTrainer):
    def __init__(self,
                 client_id,
                 dataset,
                 partition,
                 num_labels,
                 args,
                 net,
                 malicious
                 ):
        super(SyncClient, self).__init__(client_id, dataset, partition, num_labels, args, net, malicious)


class EncryptedClient(AbstractTrainer):
    def __init__(self,
                 client_id,
                 dataset,
                 partition,
                 num_labels,
                 args,
                 net,
                 malicious,
                 context=None
                 ):
        super(EncryptedClient, self).__init__(client_id, dataset, partition, num_labels, args, net, malicious)
        # 加密上下文
        self.context = context

    def set_parameters(self, global_parameters):
        # 如果加密上下文不为null，则需要解密
        parameters = copy.deepcopy(global_parameters)
        if self.context is not None:
            for level, param in parameters.items():
                parameters[level] = torch.tensor(param.decrypt().tolist(), device=self.args.device)
        self.net.load_state_dict(parameters)

    def get_parameters(self):
        local_update = super().get_parameters()
        # 加密上下文不为null，加密tensor
        if self.context is not None:
            for level, param in local_update.items():
                local_update[level] = ts.ckks_tensor(self.context, local_update[level])
        return local_update


if __name__ == '__main__':
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    tensor_1 = torch.tensor([[1, 2], [3, 4]])
    tensor_2 = torch.tensor([[3, 4], [1, 2]])
    enc_1 = ts.ckks_tensor(context, tensor_1)
    enc_2 = ts.ckks_tensor(context, tensor_2)
    enc_3 = enc_1 * enc_2
    enc_3.mul()
    print(enc_3.shape)
    print(enc_3.sum().sum().decrypt().tolist())

    print((enc_3.decrypt().tolist()))
