from torch import nn
import torch.nn.functional as F


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self) -> None:
        """
        定义网络
        """
        super(CNNMnist, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 10)
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_hidden_layers=None):
        super(MLP, self).__init__()
        if num_hidden_layers is None:
            num_hidden_layers = int(input_dim * output_dim * 2 / 3)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, num_hidden_layers),
            nn.ReLU(),
            nn.Linear(num_hidden_layers, output_dim)
        )

    def forward(self, x):
        return self.model(x)
