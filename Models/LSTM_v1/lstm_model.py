import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from sklearn.metrics import f1_score, classification_report, confusion_matrix

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
import torch.nn as nn

class LSTMWindowClassifier(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=1, num_classes=14, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Group the fully connected layers here. 
        # You can now easily swap, add, or remove layers.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # last layer's hidden state
        
        # Pass the hidden state through the sequential block
        out = self.classifier(last_hidden)
        return out
    
def predict(model, test_folder, label_encoder, window_size=5, batch_size=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Load the test data using your existing Dataset class
        test_dataset = PoseWindowDataset(test_folder, window_size=window_size, label_encoder=label_encoder)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        y_true = []
        y_pred = []

        print(f"Evaluating on {len(test_dataset)} test windows...")

        with torch.no_grad():
            for seq, label in test_loader:
                seq = seq.to(device)
                
                outputs = model(seq)
                preds = torch.argmax(outputs, dim=1)
                
                y_true.extend(label.numpy())
                y_pred.extend(preds.cpu().numpy())

        return y_true, y_pred

# -------------------------
# Training
# -------------------------
def patient_train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    label_encoder,
    encoder_path,
    model_path,
    max_epochs=50,
    patience=5
):
    
    train_hist = []
    val_hist = []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):

        # ================================
        # 1. TRAINING LOOP
        # ================================
        model.train()
        running_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # ================================
        # 2. VALIDATION LOOP
        # ================================
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        train_hist.append(epoch_train_loss)
        val_hist.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # ================================
        # 3. EARLY STOPPING
        # ================================
        if epoch_val_loss < best_val_loss:
            
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved to {model_path}")
            
        else:
            epochs_no_improve += 1
            print(f"Val loss did not improve ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # ================================
    # 4. SAVE LABEL ENCODER
    # ================================
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\nTraining complete!")
    print(f"Best model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")
    
    return model, train_hist, val_hist

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

    # ==========================
    # 1. Paths
    # ==========================
    train_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Train_clips_augmented"
    test_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Test_clips_augmented"

    model_output = "Models/LSTM_v1/phase2LSTM.pth"
    encoder_path = "Models/LSTM_v1/label_encoder.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "test"   # "train" or "test"

    # ==========================
    # 2. Data Preparation
    # ==========================
    window_size = 3
    batch_size = 128

    print("Loading training data...")
    train_dataset = PoseWindowDataset(train_folder, window_size=window_size)

    print("Loading validation data...")
    val_dataset = PoseWindowDataset(
        test_folder,
        window_size=window_size,
        label_encoder=train_dataset.label_encoder
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.label_encoder.classes_)
    print(f"\n✅ Ready! Classes ({num_classes}): {train_dataset.label_encoder.classes_}")

    # ==========================
    # 3. MODEL
    # ==========================
    model = LSTMWindowClassifier(
        input_size=84,
        hidden_size=64,
        num_layers=1,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ==========================
    # 4. TRAIN OR LOAD MODEL
    # ==========================
    if mode == "train":

        model, train_hist, val_hist = patient_train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            train_dataset.label_encoder,
            encoder_path = encoder_path,
            model_path= model_output,
            max_epochs=70,
            patience=5
        )

        label_encoder = train_dataset.label_encoder

    else:

        print("Loading label encoder...")
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        print("Loading trained model weights...")
        model.load_state_dict(torch.load(model_output, map_location=device))
        model.eval()

    # ==========================
    # 5. Evaluate Model
    # ==========================
    trues, preds = predict(model, test_folder, label_encoder)

    print("\n--- TEST SET PERFORMANCE ---")

    cm = confusion_matrix(trues, preds)
    print(cm)

    report_dict = classification_report(
        trues,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df)

    # ==========================
    # 6. Plot Classification Report
    # ==========================
    cdict = {
    "red":   ((0.0, 0.5, 0.5),   # dark red at 0
              (0.5, 1.0, 1.0),   # lighter in the middle
              (1.0, 0.0, 0.0)),  # dark green at 1
    "green": ((0.0, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.5, 0.5)),
    "blue":  ((0.0, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0))
}

    custom_cmap = LinearSegmentedColormap("RedLightGreen", segmentdata=cdict)

    metrics_df = report_df.drop(columns=["support"])

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        metrics_df,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        vmin=0, vmax=1  # force scale 0-1
    )

    # Draw a line under 'accuracy'
    divider_index = metrics_df.index.get_loc("accuracy")
    ax.hlines(divider_index, *ax.get_xlim(), colors="black", linewidth=2)

    plt.title("Classification Report LSTM")
    plt.tight_layout()
    plt.savefig("classification_report_lstm.png", dpi=300)
    plt.show()