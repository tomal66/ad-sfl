"""Reference dataset preparation for AD-SFL anomaly detection.

Creates a reference dataset by holding out a portion of the original training data.
"""

from typing import Tuple, List, Optional
import numpy as np
from torch.utils.data import Dataset, Subset

def create_reference_dataset(
    dataset: Dataset, 
    reference_size: int, 
    stratify: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into a new training set and a reference set.
    The reference set is held by the server in AD-SFL to profile normal behaviour.
    
    Args:
        dataset: The original PyTorch training dataset
        reference_size: Exact number of samples for the reference set
        stratify: Whether to maintain class distribution in the reference set
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (new_train_dataset, reference_dataset)
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()
        
    total_size = len(dataset)
    
    if reference_size >= total_size:
        raise ValueError(f"reference_size ({reference_size}) must be smaller than dataset size ({total_size})")

    if not stratify:
        # Simple random split
        indices = rng.permutation(total_size)
        ref_indices = indices[:reference_size]
        train_indices = indices[reference_size:]
        return Subset(dataset, train_indices), Subset(dataset, ref_indices)
        
    # Stratified split
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, 'targets'):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
        
        # When accessing dataset.dataset.targets via Subset indices, it might return a PyTorch tensor 
        # or list. Ensure we have a clean numpy array here.
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
    else: # Slow fallback
        targets = np.array([y for _, y in dataset])
        
    classes, class_counts = np.unique(targets, return_counts=True)
    num_classes = len(classes)
    
    # Calculate how many samples we need per class
    ref_proportions = class_counts / total_size
    ref_samples_per_class = np.floor(ref_proportions * reference_size).astype(int)
    
    # Distribute remaining samples to reach exactly reference_size
    remaining = reference_size - ref_samples_per_class.sum()
    if remaining > 0:
        added_classes = rng.choice(num_classes, size=remaining, replace=False)
        for c in added_classes:
            ref_samples_per_class[c] += 1
            
    ref_indices = []
    train_indices = []
    
    for idx, c in enumerate(classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        
        n_ref = ref_samples_per_class[idx]
        ref_indices.extend(class_indices[:n_ref])
        train_indices.extend(class_indices[n_ref:])
        
    return Subset(dataset, train_indices), Subset(dataset, ref_indices)
