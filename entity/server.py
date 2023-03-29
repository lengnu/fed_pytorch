import copy

from aggregate.abstract import AbstractAggregator
from aggregate.fed_avg import FedAvgAggregator
from aggregate.krum import KrumAggregator
from aggregate.median import MedianAggregator
from aggregate.sided_discard import SidedDiscardAggregator, CKKSSidedDiscardAggregator
from aggregate.trimmed_mean import TrimmedMeanAggregator
from entity.abstract import AbstractServer
import tenseal as ts


class GeneralServer(AbstractServer):
    def __init__(self, args, init_parameters, context=None):
        super().__init__(args, init_parameters, context)

    def _before_aggregate(self, client_updates, **kwargs):
        pass

    def _post_aggregate(self, client_updates, **kwargs):
        pass

    def _find_aggregator(self, args) -> AbstractAggregator:
        if args.strategy == 'fed_avg':
            return FedAvgAggregator(args)
        elif args.strategy == 'krum':
            return KrumAggregator(args)
        elif args.strategy == 'median':
            return MedianAggregator(args)
        elif args.strategy == 'trimmed_mean':
            return TrimmedMeanAggregator(args)
        elif args.strategy == 'sided_discard':
            return SidedDiscardAggregator(args)
        else:
            raise ValueError('There is no such aggregation strategy ', args.strategy)


class CKKSServer(GeneralServer):
    def __init__(self, args, init_parameters, context):
        super().__init__(args, init_parameters, context)

    def _find_aggregator(self, args) -> AbstractAggregator:
        return CKKSSidedDiscardAggregator(args, self.context)

    def init_params(self, init_parameters):
        global_parameters = copy.deepcopy(init_parameters)
        for neural_level, param in global_parameters.items():
            global_parameters[neural_level] = ts.ckks_tensor(self.context, param.cpu().flatten())
        return global_parameters
