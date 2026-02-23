"""Dataset structures and loaders for AD-SFL experiments.

Supported datasets: MNIST, CIFAR10, CIFAR100, ImageNet.
"""

from typing import Tuple, Optional
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os

def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Get standard data transformations for the specified dataset."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif dataset_name in ['cifar10', 'cifar100']:
        if dataset_name == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Load train and test datasets.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100', 'imagenet')
        data_dir: Directory to download/load the data from
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dataset_name = dataset_name.lower()
    os.makedirs(data_dir, exist_ok=True)
    
    train_transform = get_transforms(dataset_name, train=True)
    test_transform = get_transforms(dataset_name, train=False)
    
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'imagenet':
        # ImageNet requires manual download. We use ImageFolder.
        train_dir = os.path.join(data_dir, 'imagenet', 'train')
        val_dir = os.path.join(data_dir, 'imagenet', 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(f"ImageNet dataset not found in {data_dir}/imagenet. "
                                  "Please download and extract it manually.")
            
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        test_dataset = datasets.ImageFolder(val_dir, test_transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return train_dataset, test_dataset
