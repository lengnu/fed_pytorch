import sys
import os


def get_root_path():
    current_file_path = os.getcwd()
    project_root_path = None
    # print("current_file_path:\t" + current_file_path)
    index = 0
    for path in sys.path:
        # print("sys_path%s:\t\t\t" % index + path)
        index += 1
        if current_file_path == path:
            continue

        if current_file_path.__contains__(path):
            project_root_path = path
            break

    if project_root_path is None:
        # 如果未获取到，说明当前路径为根路径
        project_root_path = current_file_path
        # 替换斜杠
        project_root_path = project_root_path.replace("\\", "/")
    return project_root_path


MNIST_DATA_PATH = get_root_path() + '/dataset/mnist/data'
MNIST_SPLIT_PATH = get_root_path() + '/dataset/mnist/split'

CIFAR10_DATA_PATH = get_root_path() + '/dataset/cifar10/data'
CIFAR10_SPLIT_PATH = get_root_path() + '/dataset/cifar10/split'


RESULTS_PATH = get_root_path() + '/result'
