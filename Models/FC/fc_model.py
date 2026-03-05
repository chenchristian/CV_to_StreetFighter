import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import glob
import joblib

class DynamicFCBaseline(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        
        # We will store all our layers in this list
        layers = []
        
        # Track the input size as we build layers
        current_input_size = input_size
        
        # Dynamically stack the requested number of hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # After the first layer, all subsequent hidden layers take 'hidden_size' as input
            current_input_size = hidden_size 
            
        # nn.Sequential unpacks the list and chains them together automatically
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final output layer (no Softmax, just raw logits)
        self.out = nn.Linear(current_input_size, num_classes)

    def forward(self, x):
        # x shape coming in: (batch_size, 84)
        
        # Pass through all dynamic hidden layers
        x = self.feature_extractor(x)
        
        # Pass through the final output layer
        out = self.out(x) 
        return out



def load_flat_csv_data(train_folder_path, test_folder_path, batch_size=64):
    print("Gathering CSV files from folders...")
    
    def read_folder(folder_path):
        # Finds all .csv files in the folder
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        
        # Read and combine them
        df_list = [pd.read_csv(f) for f in all_files]
        combined_df = pd.concat(df_list, ignore_index=True).dropna()
        return combined_df

    # 1. Read and combine all CSVs in the folders
    train_df = read_folder(train_folder_path)
    test_df = read_folder(test_folder_path)
    
    print(f"Loaded {len(train_df)} training rows and {len(test_df)} testing rows.")

    # 2. Split into Features (X) and Labels (y)
    X_train_raw = train_df.iloc[:, :-1].values
    y_train_raw = train_df.iloc[:, -1].values
    X_test_raw = test_df.iloc[:, :-1].values
    y_test_raw = test_df.iloc[:, -1].values
    
    # 3. Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    
    # 4. Convert to Tensors
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    
    # 5. Create Loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    
    num_classes = len(label_encoder.classes_)
    print(f"✅ Ready! Found {num_classes} classes.")
    
    return train_loader, test_loader, num_classes, label_encoder

def train_and_tune(model, train_loader, val_loader, criterion, optimizer, max_epochs=30, device="cpu", patience=5):
    train_loss_history = []
    val_loss_history = []
    
    # --- Early Stopping Setup ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # 2. Validation Phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # 3. Early Stopping Logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best weights in memory
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            # Load the best weights back before finishing
            model.load_state_dict(best_model_state)
            break

    return train_loss_history, val_loss_history

import joblib # Add this to your imports at the top!

def load_full_csv_data(folder_paths, batch_size=64):
    """Loads and combines all CSVs from multiple folders into one massive training set."""
    print(f"Gathering CSV files from folders: {folder_paths}...")
    
    all_files = []
    for folder in folder_paths:
        files = glob.glob(os.path.join(folder, "*.csv"))
        if not files:
            print(f"Warning: No CSVs found in {folder}")
        all_files.extend(files)
        
    if not all_files:
        raise FileNotFoundError("No CSV files found in any of the provided folders.")
    
    # Read and combine them
    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True).dropna()
    
    print(f"Loaded a massive dataset of {len(combined_df)} rows for final training.")

    # Split into Features (X) and Labels (y)
    X_raw = combined_df.iloc[:, :-1].values
    y_raw = combined_df.iloc[:, -1].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # Create one single DataLoader
    full_dataset = TensorDataset(X_tensor, y_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    num_classes = len(label_encoder.classes_)
    print(f"✅ Ready! Found {num_classes} classes.")
    
    return full_loader, num_classes, label_encoder


def train_final_model(model, data_loader, criterion, optimizer, epochs, device="cpu"):
    """Trains the model on the full dataset without validation or early stopping."""
    print(f"Starting final training for {epochs} epochs...")
    model.train() # Set model to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("✅ Final training complete!")
    return model


def save_model_for_live(model, label_encoder, model_path="street_fighter_fc_weights.pth", le_path="label_encoder.pkl"):
    """Saves the weights and the encoder to your hard drive."""
    # 1. Save the PyTorch Model Weights
    torch.save(model.state_dict(), model_path)
    
    # 2. Save the Label Encoder mapping
    joblib.dump(label_encoder, le_path)
    
    print(f"💾 Model weights saved to: {model_path}")
    print(f"💾 Label encoder saved to: {le_path}")



if __name__ == "__main__":
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Final Model Build on {device}...")

    # 2. Define your data folders (Combining Train and Test for 100% data usage)
    train_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Train_clips_augmented"
    test_folder = "Data/Train_Test_Data/Not_Seperated/Clips_Split_80_20/Test_clips_augmented"
    folders_to_combine = [train_folder, test_folder]

    # 3. Load the massive combined dataset
    try:
        full_loader, num_classes, label_encoder = load_full_csv_data(
            folders_to_combine, 
            batch_size=64
        )
    except FileNotFoundError as e:
        print(f"❌ Path Error: {e}")
        print("Make sure you are running this script from the correct root directory!")
        exit(1)

    # 4. Initialize the Model
    final_model = DynamicFCBaseline(
        input_size=84,       # Make sure this matches your 84 landmarks
        hidden_size=128, 
        num_layers=2, 
        num_classes=num_classes, 
        dropout=0.3
    ).to(device)

    # 5. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    # 6. Train the Model 
    # (Set this to the optimal number of epochs you found in your notebook, e.g., 34)
    optimal_epochs = 34 
    final_model = train_final_model(
        final_model, 
        full_loader, 
        criterion, 
        optimizer, 
        epochs=optimal_epochs, 
        device=device
    )

    # 7. Save for Live Play!
    save_model_for_live(
        final_model, 
        label_encoder, 
        model_path="Models/FC/street_fighter_fc_weights.pth", 
        le_path="Models/FC/label_encoder.pkl"
    )
    
    print("🎮 Ready for the Street Fighter live script!")