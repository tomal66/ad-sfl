"""LeNet model implementation for SplitFed/AD-SFL.

Defines the Client and Server portions of LeNet-5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClientLeNet(nn.Module):
    """
    Client-side LeNet up to the first pooling layer.
    Input: (B, 1, 28, 28)
    Output: (B, 6, 14, 14)
    """
    def __init__(self):
        super(ClientLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x


class ServerLeNet(nn.Module):
    """
    Server-side LeNet from the second convolution block to the output.
    Input: (B, 6, 14, 14)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super(ServerLeNet, self).__init__()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
