import torch
import numpy as np

class FCLivePosePredictor:
    """
    Real-time prediction for a Fully Connected model.
    Processes a single frame at a time and returns the winning class 
    and the full probability distribution.
    """
    def __init__(self, model, label_encoder, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval() # Ensure dropout is turned off for inference!
        self.label_encoder = label_encoder
        # Cache the class names for faster lookup
        self.classes = self.label_encoder.classes_

    def predict(self, pose_vector):
        """
        Processes a single new frame and returns the top prediction and all probabilities.
        
        Args:
            pose_vector: List or numpy array of 84 features (from a single frame).
            
        Returns:
            pred_label (str): The predicted class name (e.g., "PUNCH")
            probs (np.array): Array of probabilities for all classes
        """
        # 1. Prepare tensor: Shape becomes (Batch=1, Features=84)
        # .unsqueeze(0) adds the batch dimension that PyTorch expects
        tensor_data = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Forward pass
        with torch.no_grad():
            logits = self.model(tensor_data)
            
            # Softmax turns raw scores into 0.0-1.0 probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # 3. Find the winning class
            pred_idx = np.argmax(probs)
            pred_label = self.classes[pred_idx]

        return pred_label, probs