import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Custom CNN for landmark classification (50 classes).
    
    Architecture: 5 conv blocks with BatchNorm + 2 fully connected layers.
    Progressive channel growth: 16 -> 32 -> 64 -> 128 -> 256
    """
    
    def __init__(self, num_classes=50, dropout=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 16, output: 112x112
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16 -> 32, output: 56x56
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32 -> 64, output: 28x28
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 64 -> 128, output: 14x14
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 128 -> 256, output: 7x7
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
