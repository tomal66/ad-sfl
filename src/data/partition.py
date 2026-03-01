import numpy as np
from torch.utils.data import Subset

def _get_targets(dataset):
    # Works for torchvision datasets and your HFWrapperDataset
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    # HFWrapperDataset: labels are in the wrapped hf_dataset
    if hasattr(dataset, "hf_dataset") and hasattr(dataset, "label_key"):
        return np.asarray(dataset.hf_dataset[dataset.label_key])
    # Slow fallback (avoid if possible)
    return np.asarray([y for _, y in dataset])

def partition_data_iid(dataset, num_clients, seed=None):
    """
    IID partition that uses ALL samples (no dropping remainder).
    """
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(dataset))
    rng.shuffle(idxs)

    # Split as evenly as possible
    splits = np.array_split(idxs, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, num_classes=None, seed=None):
    """
    Non-IID Dirichlet partition that uses ALL samples.
    Automatically infers num_classes if not provided.
    """
    rng = np.random.default_rng(seed)
    targets = _get_targets(dataset)

    if num_classes is None:
        # assumes labels are 0..C-1; works for CIFAR-10/100, TinyImageNet, etc.
        num_classes = int(targets.max()) + 1

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_k = np.where(targets == c)[0]
        rng.shuffle(idx_k)

        # Dirichlet proportions for class c
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Turn proportions into counts that sum exactly to len(idx_k)
        counts = np.floor(proportions * len(idx_k)).astype(int)
        diff = len(idx_k) - counts.sum()
        if diff > 0:
            # distribute remainder to the largest fractional parts (more stable than random)
            frac = proportions * len(idx_k) - np.floor(proportions * len(idx_k))
            for i in np.argsort(-frac)[:diff]:
                counts[i] += 1

        splits = np.split(idx_k, np.cumsum(counts)[:-1])
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    # Build subsets
    client_datasets = []
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        client_datasets.append(Subset(dataset, client_indices[i]))

    return client_datasets