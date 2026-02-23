"""Partition algorithms for federated learning.

Includes IID and Dirichlet (Non-IID) partitioning strategies.
"""

from typing import List, Dict, Union, Tuple
import numpy as np
from torch.utils.data import Dataset, Subset

def partition_iid(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    Partition the dataset evenly across clients in an IID fashion.
    
    Args:
        dataset: PyTorch Dataset to partition
        num_clients: Number of partitions/clients
        
    Returns:
        List of PyTorch Subset objects
    """
    num_items = int(len(dataset) / num_clients)
    
    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)
    
    subsets = []
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = (i + 1) * num_items if i != (num_clients - 1) else len(dataset)
        client_idxs = all_idxs[start_idx:end_idx]
        subsets.append(Subset(dataset, client_idxs))
        
    return subsets


def partition_dirichlet(
    dataset: Dataset, 
    num_clients: int, 
    alpha: float, 
    num_classes: int = None
) -> List[Subset]:
    """
    Partition the dataset using a Dirichlet distribution over classes (Non-IID).
    
    Args:
        dataset: PyTorch Dataset to partition
        num_clients: Number of partitions/clients
        alpha: Concentration parameter of the Dirichlet distribution. 
               Lower means more non-IID.
        num_classes: Number of distinct classes (auto-inferred if None)
        
    Returns:
        List of PyTorch Subset objects
    """
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, 'targets'):
        # Handle the case where dataset is a Subset
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        # Slow fallback: iterate through the dataset
        targets = np.array([y for _, y in dataset])
        
    if num_classes is None:
        num_classes = len(np.unique(targets))
        
    min_size = 0
    min_require_size = 10
    
    N = len(dataset)
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance the property across clients
            prop = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(prop, idx_batch)])
            prop = prop / prop.sum()
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, prop))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
    subsets = []
    for i in range(num_clients):
        np.random.shuffle(idx_batch[i])
        subsets.append(Subset(dataset, idx_batch[i]))
        
    return subsets
