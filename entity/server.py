from typing import List, OrderedDict
from torch import Tensor

from entity.abstract import AbstractAggregator


class SyncServer(AbstractAggregator):
    def __init__(self, args, init_parameters):
        super(SyncServer, self).__init__(args, init_parameters)

    def aggregate(self, client_updates: List[OrderedDict[str, Tensor]]) -> None:
        self.global_parameters = self.aggregator(client_updates, self.args.device, self.malicious_upper)


class AsyncServer(AbstractAggregator):
    def __init__(self, args, init_parameters, context=None):
        super(AsyncServer, self).__init__(args, init_parameters)
        self.context = context





