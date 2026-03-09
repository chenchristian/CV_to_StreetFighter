import os
import glob
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class DynamicFCBaseline(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        layers = []
        current_input_size = input_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_input_size = hidden_size 
            
        self.feature_extractor = nn.Sequential(*layers)
        self.out = nn.Linear(current_input_size, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.out(x) 
        return out

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def load_and_prepare_data(train_folder, test_folder, batch_size=64):
    """Loads train and test CSVs, encodes labels, and returns DataLoaders."""
    print("Gathering and processing CSV files...")
    
    def read_folder(folder_path):
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        df_list = [pd.read_csv(f) for f in files]
        return pd.concat(df_list, ignore_index=True).dropna()

    # Read data
    train_df = read_folder(train_folder)
    test_df = read_folder(test_folder)
    print(f"Loaded {len(train_df)} training rows and {len(test_df)} testing rows.")

    # Split Features (X) and Labels (y)
    X_train_raw, y_train_raw = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    X_test_raw, y_test_raw = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values
    
    # Encode labels (Fit on BOTH to ensure all classes are captured)
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((y_train_raw, y_test_raw)))
    
    y_train_encoded = label_encoder.transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    
    # Create Loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    
    num_classes = len(label_encoder.classes_)
    print(f"✅ Data Ready! Found {num_classes} classes: {label_encoder.classes_}")
    
    return train_loader, test_loader, num_classes, label_encoder

# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_model(model, train_loader, criterion, optimizer, epochs, device="cpu"):
    """Trains the model on the training dataset."""
    print(f"\nStarting training for {epochs} epochs...")
    model.train() 
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("✅ Training complete!")
    return model

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset and returns true/predicted labels."""
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.numpy())
            
    return all_trues, all_preds

def save_model_for_live(model, label_encoder, model_path, le_path):
    """Saves the weights and the encoder to your hard drive."""
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(label_encoder, le_path)
    print(f"\n💾 Model weights saved to: {model_path}")
    print(f"💾 Label encoder saved to: {le_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # ===============================
    # 1. Setup Device
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Build on {device}...")

    mode = "test"   # "train" or "test"

    # ===============================
    # 2. Define Paths
    # ===============================
    train_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Train_clips_augmented"
    test_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Test_clips_augmented"

    model_path = "/Users/christianchen/Documents/GitHub/CV_to_StreetFighter/Models/FC/street_fighter_fc_weights.pth"
    encoder_path = "/Users/christianchen/Documents/GitHub/CV_to_StreetFighter/Models/FC/label_encoder.pkl"

    # ===============================
    # 3. Load Data
    # ===============================
    try:
        train_loader, test_loader, num_classes, label_encoder = load_and_prepare_data(
            train_folder,
            test_folder,
            batch_size=64
        )
    except FileNotFoundError as e:
        print(f"❌ Path Error: {e}")
        print("Make sure you are running this script from the correct root directory!")
        exit(1)

    # ===============================
    # 4. Initialize Model
    # ===============================
    final_model = DynamicFCBaseline(
        input_size=84,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    # ===============================
    # 5. TRAIN OR LOAD MODEL
    # ===============================
    if mode == "train":

        optimal_epochs = 34

        print("\n🔥 Training model...")

        final_model = train_model(
            final_model,
            train_loader,
            criterion,
            optimizer,
            epochs=optimal_epochs,
            device=device
        )

        # Save model + encoder
        save_model_for_live(
            final_model,
            label_encoder,
            model_path=model_path,
            le_path=encoder_path
        )

        print("\n💾 Model + encoder saved.")

    else:

        print("\n📂 Loading trained model...")

        # Load encoder
        label_encoder = joblib.load(encoder_path)

        num_classes = len(label_encoder.classes_)

        # Rebuild architecture
        final_model = DynamicFCBaseline(
            input_size=84,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(device)

        # Load weights
        final_model.load_state_dict(torch.load(model_path, map_location=device))
        final_model.eval()

        print("✅ Model loaded successfully.")

    # ===============================
    # 6. Evaluate on Test Set
    # ===============================
    print("\n--- TEST SET PERFORMANCE ---")

    trues, preds = evaluate_model(final_model, test_loader, device)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(trues, preds))

    # Classification Report
    report_dict = classification_report(
        trues,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    print("\nClassification Report:")
    print(report_df)

    # ===============================
    # 7. Plot Classification Report
    # ===============================
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

    plt.title("Classification Report FC")
    plt.tight_layout()
    plt.savefig("FC Classification report.png", dpi=300)
    plt.show()