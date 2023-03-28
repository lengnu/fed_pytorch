from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset


class DatasetSplitter(Dataset):
    def __init__(self, dataset, partition, args,malicious) -> None:
        self.dataset = dataset
        self.partition = partition
        self.args = args
        self.malicious = malicious

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index) -> T_co:
        item, label = self.dataset[self.partition[index]]
        if self.malicious:
            if self.args.label_flipping_enable:
                # 数据中毒，标签随机置换
                return item, self.num_labels - label
            if self.args.backdoor_enable and label == self.source_label:
                # 目标攻击，将某些具有特定标签的样本更改为攻击标签，对其他标签不影响
                return item, self.target_label
        return item, label
