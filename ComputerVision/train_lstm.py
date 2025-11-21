import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import glob
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


class PoseSequenceDataset(Dataset):
    """PyTorch Dataset for pose sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMPoseClassifier(nn.Module):
    """LSTM model for pose sequence classification."""
    
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
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # Take the last timestep
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def load_data(data_dir="Data", pattern="*_labeled*.csv"):
    """
    Load all labeled CSV files from the data directory.
    
    Args:
        data_dir: Directory containing labeled CSV files
        pattern: File pattern to match (default: files ending with _labeled*.csv)
    
    Returns:
        X: List of pose sequences (each sequence is a numpy array)
        y: List of labels
    """
    csv_files = glob.glob(os.path.join(data_dir, pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir} matching pattern {pattern}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    all_sequences = []
    all_labels = []
    
    print(f"\n[STEP 1] Processing {len(csv_files)} CSV files...")
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"  [{idx}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}", end=" ... ")
        
        df = pd.read_csv(csv_file)
        
        if "label" not in df.columns:
            print("SKIPPED (no 'label' column)")
            continue
        
        # Extract pose features (all columns except 'label')
        feature_cols = [col for col in df.columns if col != "label"]
        pose_data = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values
        
        # Create sequences from consecutive frames
        sequences, seq_labels = create_sequences(pose_data, labels, sequence_length=20)
        
        all_sequences.extend(sequences)
        all_labels.extend(seq_labels)
        print(f"✓ {len(sequences)} sequences extracted")
    
    print(f"\n[STEP 1 COMPLETE] Total sequences created: {len(all_sequences)}")
    print(f"  Label distribution: {pd.Series(all_labels).value_counts().to_dict()}")
    
    return np.array(all_sequences), np.array(all_labels)


def create_sequences(pose_data, labels, sequence_length=20, stride=1):
    """
    Create sequences of consecutive frames for LSTM training.
    
    Args:
        pose_data: Array of pose vectors (num_frames, num_features)
        labels: Array of labels (num_frames,)
        sequence_length: Number of frames per sequence
        stride: Step size for sliding window (1 = every frame, 2 = every other frame)
    
    Returns:
        sequences: Array of shape (num_sequences, sequence_length, num_features)
        sequence_labels: Array of shape (num_sequences,) - label for the last frame in sequence
    """
    sequences = []
    sequence_labels = []
    
    for i in range(0, len(pose_data) - sequence_length + 1, stride):
        seq = pose_data[i:i + sequence_length]
        # Use the label of the last frame in the sequence
        label = labels[i + sequence_length - 1]
        sequences.append(seq)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu', 
                model_save_path="Model/pose_lstm_model.pth"):
    """Train the LSTM model."""
    
    print("  Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"    ✓ Loss function: CrossEntropyLoss")
    print(f"    ✓ Optimizer: Adam (lr=0.001)")
    print(f"    ✓ Learning rate scheduler: ReduceLROnPlateau")
    print(f"    ✓ Early stopping: patience=10 epochs")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n  Training configuration:")
    print(f"    Device: {device}")
    print(f"    Max epochs: {num_epochs}")
    print(f"    Model save path: {model_save_path}")
    print(f"    Initial learning rate: {current_lr}")
    print(f"\n  Starting training...\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}", end="")
        if current_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ NEW BEST MODEL! Saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience} epochs without improvement)")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"  No improvement for {patience} consecutive epochs")
            break
    
    # Load best model
    print(f"\n  Loading best model from checkpoint...")
    model.load_state_dict(torch.load(model_save_path))
    print(f"  ✓ Best model loaded (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
    print(f"  ✓ Model saved to: {model_save_path}")
    
    return model


def evaluate_model(model, test_loader, label_encoder, device):
    """Evaluate the trained model."""
    print("  Running evaluation on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = validate(model, test_loader, criterion, device)
    
    print(f"\n  Test Results:")
    print(f"    Test Loss: {test_loss:.4f}")
    print(f"    Test Accuracy: {test_acc:.2f}%")
    
    # Decode labels
    print(f"\n  Decoding predictions...")
    y_test_decoded = label_encoder.inverse_transform(all_labels)
    y_pred_decoded = label_encoder.inverse_transform(all_preds)
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    print(cm)
    
    # Per-class accuracy
    print(f"\n  Per-class Performance:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = (y_test_decoded == class_name)
        if class_mask.sum() > 0:
            class_acc = (y_pred_decoded[class_mask] == class_name).sum() / class_mask.sum() * 100
            print(f"    {class_name:8s}: {class_acc:.2f}% ({class_mask.sum()} samples)")


def main():
    print("="*50)
    print("LSTM Pose Classification Training (PyTorch)")
    print("="*50)
    
    # Find project root directory (parent of ComputerVision/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)  # Change to project root so paths work correctly
    print(f"Project root: {project_root}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("[STEP 1] LOADING DATA")
    print("="*50)
    
    # Load from Data folder (original labeled data)
    print(f"\n[1.1] Loading data from Data/ folder...")
    X, y = load_data(data_dir="Data", pattern="*_labeled*.csv")
    
    # Encode labels
    print(f"\n[STEP 2] ENCODING LABELS")
    print("="*50)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"  Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print(f"  Total samples: {len(y_encoded)}")
    
    # Split data
    print(f"\n[STEP 3] SPLITTING DATA")
    print("="*50)
    print("  Splitting into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  ✓ Train: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  ✓ Validation: {len(X_val)} samples ({100*len(X_val)/len(X):.1f}%)")
    print(f"  ✓ Test: {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
    
    # Create datasets and dataloaders
    print(f"\n[STEP 4] CREATING DATASETS AND DATALOADERS")
    print("="*50)
    print("  Creating PyTorch datasets...")
    train_dataset = PoseSequenceDataset(X_train, y_train)
    val_dataset = PoseSequenceDataset(X_val, y_val)
    test_dataset = PoseSequenceDataset(X_test, y_test)
    print("  ✓ Datasets created")
    
    print("  Creating data loaders (batch_size=32)...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Validation batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # Model parameters
    print(f"\n[STEP 5] BUILDING MODEL")
    print("="*50)
    input_size = X_train.shape[2]  # Number of features per frame
    num_classes = len(np.unique(y_encoded))
    sequence_length = X_train.shape[1]
    
    print(f"  Analyzing data shape...")
    print(f"    Input size: {input_size} features per frame")
    print(f"    Sequence length: {sequence_length} frames")
    print(f"    Number of classes: {num_classes}")
    
    print(f"\n  Creating LSTM model...")
    model = LSTMPoseClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    print("\n  Model Architecture:")
    print(model)
    
    # Train model
    print(f"\n[STEP 6] TRAINING MODEL")
    print("="*50)
    model = train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    
    # Evaluate
    print(f"\n[STEP 7] EVALUATING MODEL")
    print("="*50)
    evaluate_model(model, test_loader, label_encoder, device)
    
    # Save label encoder
    print(f"\n[STEP 8] SAVING FILES")
    print("="*50)
    import pickle
    encoder_path = "Model/label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Label encoder saved to: {encoder_path}")
    print(f"  ✓ Model already saved during training")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
