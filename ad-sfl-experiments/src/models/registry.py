"""Factory to get model by name.

Provides functions to construct Client and Server models depending on the dataset/architecture.
"""

import torch.nn as nn
from .lenet import ClientLeNet, ServerLeNet
from .resnet18 import ClientResNet18, ServerResNet18

def get_client_model(model_name: str) -> nn.Module:
    """
    Get the client-side model portion.
    
    Args:
        model_name: Name of the architecture ('lenet', 'resnet18')
        
    Returns:
        Instantiated PyTorch module for the client.
    """
    model_name = model_name.lower()
    if model_name == 'lenet':
        return ClientLeNet()
    elif model_name == 'resnet18':
        return ClientResNet18(in_channels=3)
    else:
        raise ValueError(f"Unknown client model architecture: {model_name}")


def get_server_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Get the server-side model portion.
    
    Args:
        model_name: Name of the architecture ('lenet', 'resnet18')
        num_classes: Number of output classes for the final layer
        
    Returns:
        Instantiated PyTorch module for the server.
    """
    model_name = model_name.lower()
    if model_name == 'lenet':
        return ServerLeNet(num_classes=num_classes)
    elif model_name == 'resnet18':
        return ServerResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown server model architecture: {model_name}")
