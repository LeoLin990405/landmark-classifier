import torch
import torch.nn as nn
import torch.optim as optim


def get_loss():
    """Return the loss function for training."""
    return nn.CrossEntropyLoss()


def get_optimizer(model, learning_rate=0.001, weight_decay=1e-5):
    """Return the optimizer for training.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization weight decay
    
    Returns:
        Adam optimizer configured for the model parameters
    """
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
