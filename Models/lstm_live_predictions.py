import torch
import numpy as np
from collections import deque

class LivePosePredictor:
    """
    Real-time prediction using a trained LSTM model and pose vectors from PoseWorker.
    """
    def __init__(self, model, label_encoder, sequence_length=5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.label_encoder = label_encoder
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)

    def predict(self, pose_vector):
        """
        Push a new pose vector, predict action if buffer is full.
        Args:
            pose_vector: np.array of shape (84,) for 21 landmarks * 4 features
        Returns:
            (pred_label, confidence) or (None, None) if buffer not full yet
        """
        # Append new frame to buffer
        self.frame_buffer.append(np.array(pose_vector, dtype=np.float32))

        # Wait until we have enough frames
        if len(self.frame_buffer) < self.sequence_length:
            return None, None

        # Convert buffer to tensor
        seq_tensor = torch.from_numpy(np.array(self.frame_buffer, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_label = self.label_encoder.inverse_transform([pred_idx.item()])[0]
            conf = conf.item()

        return pred_label, conf
