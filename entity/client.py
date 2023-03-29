import copy

import torch

from entity.abstract import AbstractTrainer
import tenseal as ts


class GeneralClient(AbstractTrainer):
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
        super(GeneralClient, self).__init__(client_id, dataset, partition, num_labels, args, net, malicious, context)


class CKKSClient(AbstractTrainer):
    def __init__(self,
                 client_id,
                 dataset,
                 partition,
                 num_labels,
                 args,
                 net,
                 malicious,
                 context
                 ):
        super(CKKSClient, self).__init__(client_id, dataset, partition, num_labels, args, net, malicious, context)
        # 加密上下文
        if context is None:
            raise ValueError('enc_context can not be None')

    def set_parameters(self, global_parameters):
        # 解密CKKSTensor并变化为net的形状
        parameters = copy.deepcopy(global_parameters)
        for neural_level, param in parameters.items():
            parameters[neural_level] = torch.tensor(param.decrypt().tolist(), device=self.args.device,
                                                    dtype=torch.float).reshape(self.net_meta[neural_level])
        self.net.load_state_dict(parameters)

    def get_parameters(self):
        local_update = super().get_parameters()
        # 加密Tensor
        for neural_level, param in local_update.items():
            local_update[neural_level] = ts.ckks_tensor(self.enc_context, local_update[neural_level].flatten())
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
    print(torch.tensor(enc_1.decrypt().tolist()).reshape(2, 2))
