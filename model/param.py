"""
@FileName：param.py
@Description：
@Author：duwei
@Time：2023/3/28 14:54
@Email: 1456908874@qq.com
"""
from abc import ABC


class AbstractParameters(ABC):
    """
    抽象参数类
    """

    def __init__(self, **kwargs):
        self.check(**kwargs)
        self.kwargs = kwargs
        self.meta = kwargs['meta']
        self.params = kwargs['params']

    def check(self, **kwargs):
        if kwargs['meta'] is None:
            raise ValueError('Metadata cannot be empty')
        if kwargs['params'] is None:
            raise ValueError('Params cannot be empty')

    def get_params(self):
        return self.params
