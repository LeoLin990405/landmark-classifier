import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Train the model for one epoch.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, data_loader, criterion, device):
    """Validate the model on the validation/test set.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, data_loaders, optimizer, criterion, scheduler, device,
                n_epochs=30, save_path="best_model.pth"):
    """Full training loop with validation and model checkpointing.
    
    Args:
        model: The neural network model
        data_loaders: Dict with 'train' and 'valid' DataLoaders
        optimizer: The optimizer
        criterion: The loss function
        scheduler: Learning rate scheduler (ReduceLROnPlateau)
        device: torch device
        n_epochs: Number of training epochs
        save_path: Path to save the best model
    
    Returns:
        dict: Training history with losses and accuracies
    """
    history = {
        "train_loss": [], "train_acc": [],
        "valid_loss": [], "valid_acc": [],
    }
    best_valid_acc = 0.0
    
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, data_loaders["train"], optimizer, criterion, device
        )
        valid_loss, valid_acc = validate(
            model, data_loaders["valid"], criterion, device
        )
        
        scheduler.step(valid_loss)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        
        print(f"Epoch {epoch}/{n_epochs} — "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (valid acc: {valid_acc:.2f}%)")
    
    print(f"\nBest validation accuracy: {best_valid_acc:.2f}%")
    return history


def test_model(model, data_loader, device):
    """Test the model and return accuracy.
    
    Returns:
        float: Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_training_history(history):
    """Plot training and validation loss/accuracy curves."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["valid_loss"], label="Valid Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["valid_acc"], label="Valid Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
