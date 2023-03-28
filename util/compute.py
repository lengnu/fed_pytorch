from torch import Tensor


def euclidean_distance(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    计算两个tensor之间欧式距离
    :param tensor_1:   tensor1
    :param tensor_2:   tensor2
    :return:    欧氏距离
    """

    return (tensor_1 - tensor_2).pow(2).sum()
