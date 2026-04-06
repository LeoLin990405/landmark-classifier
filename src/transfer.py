import torch
import torch.nn as nn
from torchvision import models


def get_pretrained_model(num_classes=50):
    """Return a pretrained ResNet50 model with a new classification head.
    
    Strategy:
    - Load ResNet50 pretrained on ImageNet
    - Freeze all existing layers
    - Replace the final fc layer with Linear(2048, num_classes)
    - Only the new classification head will be trained
    
    Args:
        num_classes: Number of output classes (50 landmarks)
    
    Returns:
        ResNet50 model ready for transfer learning
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
