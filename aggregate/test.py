import collections

import torch
import tenseal as ts

import util.option
from aggregate.median import MedianAggregator
from aggregate.sided_discard import SidedDiscardAggregator, SidedDiscardEncryptAggregator
from aggregate.trimmed_mean import TrimmedMeanAggregator

if __name__ == '__main__':
    if __name__ == '__main__':
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2 ** 40

        # ts_1 = ts.ckks_tensor(context, torch.tensor([1, 2, 3, 4], dtype=torch.float))
        # ts_2 = ts.ckks_tensor(context, torch.tensor([1, 2, 3, 4], dtype=torch.float))
        # ts_1 += ts_2
        # print(ts_1.decrypt().tolist())

        clients_updates = []
        client_1_update = collections.OrderedDict()
        client_1_update['1'] = torch.zeros(2, 2, dtype=torch.float)
        client_1_update['2'] = torch.ones(2, 2, dtype=torch.float)
        client_1_update['3'] = torch.tensor([
            [[1, 2], [3, 4]],
            [[9, 7], [7.56, 2.4]],
        ], dtype=torch.float)

        client_2_update = collections.OrderedDict()
        client_2_update['1'] = torch.ones(2, 2, dtype=torch.float)
        client_2_update['2'] = torch.ones(2, 2, dtype=torch.float)
        client_2_update['3'] = torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 4], [2, 6]],
        ], dtype=torch.float)

        client_3_update = collections.OrderedDict()
        client_3_update['1'] = torch.ones(2, 2, dtype=torch.float)
        client_3_update['2'] = torch.ones(2, 2, dtype=torch.float)
        client_3_update['3'] = torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 4], [2, 6]],
        ], dtype=torch.float)

        clients_updates.append(client_1_update)
        clients_updates.append(client_2_update)
        clients_updates.append(client_3_update)
        args = util.option.args_parser()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        client__updates_ckks = []
        client_1_update_ckks = collections.OrderedDict()
        client_1_update_ckks['1'] = ts.ckks_tensor(context, torch.zeros(2, 2, dtype=torch.float).flatten())
        print(client_1_update_ckks['1'])
        client_1_update_ckks['2'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
        client_1_update_ckks['3'] = ts.ckks_tensor(context, torch.tensor([
            [[1, 2], [3, 4]],
            [[9, 7], [7.56, 2.4]],
        ], dtype=torch.float).flatten())

        client_2_update_ckks = collections.OrderedDict()
        client_2_update_ckks['1'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
        client_2_update_ckks['2'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
        client_2_update_ckks['3'] = ts.ckks_tensor(context, torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 4], [2, 6]],
        ], dtype=torch.float).flatten())

        client_3_update_ckks = collections.OrderedDict()
        client_3_update_ckks['1'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
        client_3_update_ckks['2'] = ts.ckks_tensor(context, torch.ones(2, 2, dtype=torch.float).flatten())
        client_3_update_ckks['3'] = ts.ckks_tensor(context, torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 4], [2, 6]],
        ], dtype=torch.float).flatten())

        client__updates_ckks.append(client_1_update_ckks)
        client__updates_ckks.append(client_2_update_ckks)
        client__updates_ckks.append(client_3_update_ckks)

        aggregator_1 = SidedDiscardAggregator(args)
        aggregator_2 = SidedDiscardEncryptAggregator(context, args)
        aggregator_3 = MedianAggregator(args)
        aggregator_4 = TrimmedMeanAggregator(args)
        print(aggregator_1.aggregate(clients_updates))
        print('\n\n')
        agg_ckks = aggregator_2.aggregate(client__updates_ckks)

        for level, model in agg_ckks.items():
            print('level = ', level, '\t\t model = ', model.decrypt().tolist())

        print('\n\n\n')
        print(aggregator_3.aggregate(clients_updates))
        print(aggregator_4.aggregate(clients_updates))
