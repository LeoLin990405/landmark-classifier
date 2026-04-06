import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_transforms():
    """Return data transforms for training, validation, and test sets."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    }
    return data_transforms


def get_data_loaders(data_dir, batch_size=64, num_workers=2):
    """Create data loaders for training, validation, and test sets.
    
    Args:
        data_dir: Path to the dataset root (must contain train/, valid/, test/)
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
    
    Returns:
        dict of DataLoaders keyed by 'train', 'valid', 'test'
    """
    data_transforms = get_data_transforms()
    
    image_datasets = {
        split: datasets.ImageFolder(
            os.path.join(data_dir, split),
            data_transforms[split]
        )
        for split in ["train", "valid", "test"]
    }
    
    data_loaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            image_datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    
    return data_loaders, image_datasets


def compute_mean_and_std(data_dir):
    """Compute per-channel mean and std of the dataset (before normalization)."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=2, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size
    
    mean /= n_samples
    std /= n_samples
    return mean, std


def visualize_samples(data_loader, class_names, n=5):
    """Visualize n sample images from the data loader."""
    images, labels = next(iter(data_loader))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(1, n, figsize=(20, 4))
    for i in range(n):
        img = images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
