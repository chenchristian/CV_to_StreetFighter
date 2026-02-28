import torch
import numpy as np
from collections import deque

class LivePosePredictor:
    """
    Real-time prediction returning both the winning class 
    and the full probability distribution for 14 classes.
    """
    def __init__(self, model, label_encoder, sequence_length=5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.label_encoder = label_encoder
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        # Cache the class names for faster lookup
        self.classes = self.label_encoder.classes_

    def predict(self, pose_vector):
        """
        Processes a new frame and returns the top prediction and all probabilities.
        
        Returns:
            (pred_label, all_probs_vector) 
            - pred_label: str (e.g., "PUNCH")
            - all_probs_vector: np.array of 14 floats
            - Returns (None, None) if buffer is warming up.
        """
        # 1. Add new data to sliding window
        self.frame_buffer.append(np.array(pose_vector, dtype=np.float32))

        # 2. Check if we have enough temporal context
        if len(self.frame_buffer) < self.sequence_length:
            return None, None

        # 3. Prepare tensor: (Batch=1, Seq=5, Features=84)
        seq_tensor = torch.from_numpy(np.array(self.frame_buffer, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            # Softmax turns raw scores into 0.0-1.0 probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Find the winner
            pred_idx = np.argmax(probs)
            pred_label = self.classes[pred_idx]

        return pred_label, probs