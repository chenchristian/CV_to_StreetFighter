import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2 as cv

# -------------------------
# Dataset
# -------------------------
class PoseWindowDataset(Dataset):
    def __init__(self, folder_path, window_size=5, stride=1, label_encoder=None):
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.label_encoder = label_encoder
        self.coords = [f"{axis}{i}" for i in range(21) for axis in "xyzv"]

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in {folder_path}")

        all_labels = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'label' not in df.columns:
                    continue
                df = df.dropna(subset=['label'])
                labels = df['label'].values
                all_labels.extend(labels)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)

        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'label' not in df.columns:
                    continue
                df = df.dropna(subset=['label'])
                labels = self.label_encoder.transform(df['label'])
                if len(df) < window_size:
                    continue
                data = df[self.coords].values.astype(np.float32)
                for start in range(0, len(df) - window_size + 1, stride):
                    end = start + window_size
                    window_seq = data[start:end]
                    window_label = np.bincount(labels[start:end]).argmax()
                    self.windows.append((window_seq, window_label))
            except Exception as e:
                print(f"Error processing file {f}: {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq, label = self.windows[idx]
        return torch.tensor(seq), torch.tensor(label, dtype=torch.long)

# -------------------------
# LSTM Model
# -------------------------
class LSTMWindowClassifier(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # last layer's hidden state
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# -------------------------
# Training
# -------------------------
def train_model(train_folder, window_size=5, stride=1, batch_size=16, epochs=10, lr=1e-3, model_path="lstm_pose_model.pth", encoder_path="label_encoder.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset and label encoder
    dataset = PoseWindowDataset(train_folder, window_size=window_size, stride=stride)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.label_encoder.classes_)
    model = LSTMWindowClassifier(input_size=84, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(dataset)} windows...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(dataset.label_encoder, f)

    print(f"\nTraining complete! Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    return model, dataset.label_encoder

# -------------------------
# Prediction
# -------------------------
def play_video_with_prediction(model, label_encoder, csv_path, video_path, window_size=5, stride=1, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load CSV
    df = pd.read_csv(csv_path)
    coords = [f"{axis}{i}" for i in range(21) for axis in "xyzv"]
    features = df[coords].values.astype(np.float32)
    labels = df['label'].values if 'label' in df.columns else None

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_buffer = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx >= len(features):
            break  # CSV shorter than video

        frame_feat = features[idx]
        frame_buffer.append(frame_feat)
        if len(frame_buffer) > window_size:
            frame_buffer.pop(0)

        # Predict if we have enough frames
        if len(frame_buffer) == window_size:
            seq_tensor = torch.FloatTensor(frame_buffer).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                pred_label = label_encoder.inverse_transform([pred_idx.item()])[0]
                conf = conf.item()

            # Overlay prediction
            cv.putText(frame, f"{pred_label} ({conf:.2f})", (30, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv.imshow("Prediction", frame)
        if cv.waitKey(30) & 0xFF == 27:  # ESC to quit
            break

        idx += 1

    cap.release()
    cv.destroyAllWindows()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    train_folder = "DataAugmentation/augmented_data"
    model, label_encoder = train_model(train_folder, epochs=10)

    test_csv = "pose_data.csv"
    videopath = "pose_recording.mp4"
    preds, trues = play_video_with_prediction(model, label_encoder, test_csv, video_path=videopath)

    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(trues, preds))
    print(classification_report(trues, preds))
    