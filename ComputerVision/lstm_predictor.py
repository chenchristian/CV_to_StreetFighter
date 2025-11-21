import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from collections import deque


class LSTMPoseClassifier(nn.Module):
    """LSTM model for pose sequence classification (same as training)."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMPoseClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMPosePredictor:
    """
    Real-time pose classification using trained LSTM model (PyTorch).
    """
    
    def __init__(self, model_path="Model/pose_lstm_model.pth", 
                 encoder_path="Model/label_encoder.pkl",
                 sequence_length=20,
                 device=None):
        """
        Initialize the LSTM predictor.
        
        Args:
            model_path: Path to trained LSTM model (.pth file)
            encoder_path: Path to label encoder
            sequence_length: Number of frames in sequence (must match training)
            device: Device to run on ('cpu' or 'cuda'). Auto-detects if None.
        """
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load label encoder first to get num_classes
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        num_classes = len(self.label_encoder.classes_)
        print(f"Label encoder loaded. Classes: {self.label_encoder.classes_}")
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}...")
        
        # Get feature dimension from first frame buffer initialization
        # We'll infer it from the model state dict or use a default
        # For now, we'll load a dummy to get the input_size
        dummy_input_size = 84  # Default: 21 landmarks * 4 values
        try:
            # Try to load model state to infer input_size
            state_dict = torch.load(model_path, map_location=self.device)
            # The LSTM weight_ih_l0 has shape (hidden_size*4, input_size)
            # So input_size = weight_ih_l0.shape[1]
            if 'lstm.weight_ih_l0' in state_dict:
                input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            else:
                input_size = dummy_input_size
        except:
            input_size = dummy_input_size
        
        # Create model
        self.model = LSTMPoseClassifier(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        self.feature_dim = input_size
        print(f"Model loaded successfully! Input size: {input_size}, Classes: {num_classes}")
        
        # Initialize buffer with zeros
        for _ in range(sequence_length):
            self.frame_buffer.append(np.zeros(self.feature_dim, dtype=np.float32))
    
    def predict(self, pose_vector):
        """
        Predict action from current pose vector.
        
        Args:
            pose_vector: Current frame's pose vector (array of features)
        
        Returns:
            predicted_label: Predicted action label (e.g., 'punch', 'kick', 'idle')
            confidence: Confidence score (0-1)
        """
        # Ensure pose_vector is the right shape
        if len(pose_vector) != self.feature_dim:
            raise ValueError(f"Pose vector dimension mismatch. Expected {self.feature_dim}, got {len(pose_vector)}")
        
        # Add current frame to buffer
        self.frame_buffer.append(np.array(pose_vector, dtype=np.float32))
        
        # Convert buffer to sequence tensor
        sequence = np.array(list(self.frame_buffer))
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        sequence = sequence.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequence)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
        
        predicted_class_idx = predicted_class_idx.item()
        confidence = confidence.item()
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_label, float(confidence)
    
    def reset(self):
        """Reset the frame buffer (useful when starting a new session)."""
        self.frame_buffer.clear()
        for _ in range(self.sequence_length):
            self.frame_buffer.append(np.zeros(self.feature_dim, dtype=np.float32))


def test_predictor():
    """
    Test the predictor with dummy data.
    """
    try:
        predictor = LSTMPosePredictor()
        
        # Create dummy pose vector (84 features: 21 landmarks * 4 values)
        dummy_pose = np.random.rand(84).astype(np.float32)
        
        # Need to fill buffer first
        for _ in range(20):
            label, confidence = predictor.predict(dummy_pose)
        
        print(f"\nPrediction: {label} (confidence: {confidence:.4f})")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_lstm.py")


if __name__ == "__main__":
    test_predictor()
