import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class Predictor(nn.Module):
    """TorchScript-compatible predictor for landmark classification.
    
    Wraps a trained model with preprocessing transforms for inference.
    """
    
    def __init__(self, model, class_names):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def forward(self, image_path):
        """Predict the landmark class for a given image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return self.class_names[predicted.item()], confidence.item()


def export_model(model, save_path="landmark_model.pt"):
    """Export model using TorchScript for deployment.
    
    Args:
        model: Trained PyTorch model
        save_path: Path to save the TorchScript model
    """
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    device = next(model.parameters()).device
    example_input = example_input.to(device)
    
    scripted_model = torch.jit.trace(model, example_input)
    scripted_model.save(save_path)
    print(f"Model exported to {save_path}")
    return scripted_model


def load_exported_model(model_path="landmark_model.pt"):
    """Load a TorchScript exported model.
    
    Args:
        model_path: Path to the TorchScript model file
    
    Returns:
        TorchScript model ready for inference
    """
    model = torch.jit.load(model_path)
    model.eval()
    return model
