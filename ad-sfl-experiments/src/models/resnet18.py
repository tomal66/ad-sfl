"""ResNet18 implementation for SplitFed/AD-SFL.

Defines the Client and Server portions of ResNet18 adapted for 32x32 images (CIFAR)
or 224x224 (ImageNet).
"""

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

class ClientResNet18(nn.Module):
    """
    Client-side ResNet18.
    Computes up to the end of layer1.
    """
    def __init__(self, in_channels: int = 3):
        super(ClientResNet18, self).__init__()
        # Use full resnet18 just to grab initialized layers easily
        full_resnet = resnet.resnet18()
        
        # Modify first conv if needed (e.g., for CIFAR where in_channels=3 but size=32x32)
        # Standard torchvision resnet uses 7x7 conv with stride 2, maxpool.
        # Often for CIFAR we replace it with 3x3 conv, stride 1, no maxpool.
        # We will expose a param to allow standard torchvision behavior if needed.
        if in_channels == 3:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = full_resnet.bn1
            self.relu = full_resnet.relu
            # Omit maxpool for 32x32 CIFAR, keep for ImageNet? Let's just use CIFAR setup by default.
            self.maxpool = nn.Identity() 
        else:
            self.conv1 = full_resnet.conv1
            self.bn1 = full_resnet.bn1
            self.relu = full_resnet.relu
            self.maxpool = full_resnet.maxpool

        self.layer1 = full_resnet.layer1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class ServerResNet18(nn.Module):
    """
    Server-side ResNet18.
    Computes from layer2 to the classification head.
    """
    def __init__(self, num_classes: int = 10):
        super(ServerResNet18, self).__init__()
        full_resnet = resnet.resnet18()
        
        self.layer2 = full_resnet.layer2
        self.layer3 = full_resnet.layer3
        self.layer4 = full_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * full_resnet.layer4[-1].expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
