import matplotlib.pyplot as plt
import numpy as np
import torch


def after_subplot(ax, group_name, x_label):
    """Add title and labels after creating a subplot."""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")


def plot_confusion_matrix(cm, class_names, figsize=(12, 10)):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    """Count the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
