"""
PyTorch data pipeline for the HAM10000-style skin lesion dataset.

The dataset directory (see EDA.ipynb for the full analysis) is already
split into train/validation/test folders, each containing one
sub-folder per diagnostic class:

    dataset/
        train/{akiec,bcc,bkl,df,mel,nv,vasc}/*.jpg
        validation/{...}/*.jpg
        test/{...}/*.jpg

torchvision.datasets.ImageFolder handles this layout natively and assigns
class indices alphabetically, which matches src.config.CLASS_CODES.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from src import config


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the torchvision transform pipeline for train or eval mode.

    The EDA (notebooks/EDA.ipynb, section 6) found the training split
    already contains ~2x Roboflow-baked-in augmented copies per source
    image, and heavier per-image PIL ops (rotation, color jitter) were
    measured to be the dominant per-batch cost on this CPU-only machine.
    We therefore keep only a cheap horizontal flip here on top of the
    dataset's existing augmentation, rather than stacking on more.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
    ])


def _targets_of(dataset) -> np.ndarray:
    """Works for both a plain ImageFolder and a Subset of one (as produced
    by stratified_subsample below)."""
    if isinstance(dataset, Subset):
        return np.array(dataset.dataset.targets)[dataset.indices]
    return np.array(dataset.targets)


def _classes_of(dataset):
    return dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes


def stratified_subsample(dataset: datasets.ImageFolder, fraction: float, seed: int = 42) -> Subset:
    """Return a class-stratified random subset of `dataset` at `fraction`
    of its original size (at least 1 sample per class), preserving each
    class's relative proportion.

    Used to make the multi-architecture comparison (src/training/compare_models.py)
    tractable on CPU-only machines: training on e.g. 25% of the ~14k-image
    training set still gives a directionally meaningful architecture
    ranking in a fraction of the wall-clock time. Stratifying (rather than
    a plain random sample) matters here because the dataset is ~60x
    imbalanced (see EDA.ipynb) — a flat random sample at low fractions
    could otherwise nearly wipe out the rarest class ('df').
    """
    if fraction >= 1.0:
        return Subset(dataset, list(range(len(dataset))))
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    indices = []
    for c in range(len(dataset.classes)):
        class_indices = np.where(targets == c)[0]
        n_keep = max(1, int(round(len(class_indices) * fraction)))
        indices.extend(rng.choice(class_indices, size=n_keep, replace=False).tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices)


def compute_class_weights(dataset, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights, used for the loss function and/or sampler.

    The dataset is heavily imbalanced (melanocytic nevi 'nv' outnumber
    dermatofibroma 'df' by ~60x — see EDA.ipynb), so naive training would
    collapse to predicting the majority class. We counter this with
    class-weighted cross-entropy (see src/training/train.py).
    """
    targets = _targets_of(dataset)
    class_counts = np.bincount(targets, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # avoid div-by-zero
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset, num_classes: int = config.NUM_CLASSES) -> WeightedRandomSampler:
    """Per-sample weighted sampler as an alternative/complement to loss weighting."""
    targets = _targets_of(dataset)
    class_counts = np.bincount(targets, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_datasets(train_fraction: float = 1.0):
    train_ds_full = datasets.ImageFolder(config.TRAIN_DIR, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(config.VAL_DIR, transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder(config.TEST_DIR, transform=get_transforms(train=False))

    # Sanity check: class-to-index mapping must be identical and equal to
    # the alphabetical CLASS_CODES ordering used everywhere else.
    assert train_ds_full.classes == val_ds.classes == test_ds.classes == config.CLASS_CODES, (
        f"Class ordering mismatch. train={train_ds_full.classes} "
        f"expected={config.CLASS_CODES}"
    )
    train_ds = stratified_subsample(train_ds_full, train_fraction) if train_fraction < 1.0 else train_ds_full
    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size: int = config.BATCH_SIZE, use_weighted_sampler: bool = False,
                     train_fraction: float = 1.0):
    """Build train/val/test DataLoaders.

    Args:
        train_fraction: if < 1.0, trains on a class-stratified random
            subset of that fraction of the training set (val/test are
            always full-size, so evaluation stays comparable across runs).

    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    train_ds, val_ds, test_ds = get_datasets(train_fraction=train_fraction)
    class_weights = compute_class_weights(train_ds)

    loader_kwargs = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=4 if config.NUM_WORKERS > 0 else None,
    )

    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, _classes_of(train_ds), class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes, weights = get_dataloaders()
    print("Classes:", classes)
    print("Class weights:", weights)
    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape, "Labels shape:", yb.shape)
