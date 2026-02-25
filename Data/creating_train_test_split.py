import os
import shutil
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def create_clips(file_paths, output_dir="instance_clips", pad_frames=20):
    """Extracts actions padded with up to 'pad_frames' of idle context."""
    os.makedirs(output_dir, exist_ok=True)
    counters = defaultdict(int)
    
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Extract author from the folder name (e.g., "Christian")
            folder_author = os.path.basename(os.path.dirname(file_path))
            
        except FileNotFoundError:
            print(f"  -> Error: File not found ({file_path}). Skipping.")
            continue
            
        if 'label' not in df.columns:
            print(f"  -> Skipping {file_path}: 'label' column not found.")
            continue
            
        # Find contiguous blocks of identical labels
        block_ids = (df['label'] != df['label'].shift(1)).cumsum()
        action_blocks = df[df['label'] != 'idle'].groupby(block_ids)
        
        for block_id, instance_df in action_blocks:
            start_idx = instance_df.index[0]
            end_idx = instance_df.index[-1]
            
            action_name = instance_df['label'].iloc[0]
            author = folder_author # Assign the folder name as the author
            
            # Find Left Padding
            pad_start = start_idx
            for i in range(1, pad_frames + 1):
                check_idx = start_idx - i
                if check_idx < 0:
                    break
                if df.at[check_idx, 'label'] == 'idle':
                    pad_start = check_idx
                else:
                    break 
                    
            # Find Right Padding
            pad_end = end_idx
            for i in range(1, pad_frames + 1):
                check_idx = end_idx + i
                if check_idx >= len(df):
                    break
                if df.at[check_idx, 'label'] == 'idle':
                    pad_end = check_idx
                else:
                    break 
                    
            # Extract and save the clip
            clip_df = df.iloc[pad_start:pad_end + 1]
            counters[(action_name, author)] += 1
            count = counters[(action_name, author)]
            
            filename = f"{action_name}_{author}_{count}.csv"
            out_path = os.path.join(output_dir, filename)
            clip_df.to_csv(out_path, index=False)
            
    print(f"\nFinished extracting clips to '{output_dir}/'.")

    

def split_clips(input_dir="instance_clips", train_dir="train_clips", test_dir="test_clips", test_size=0.2):
    """Performs a stratified 80/20 split based on action and author."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    clip_data = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
            
        name_without_ext = filename[:-4]
        parts = name_without_ext.rsplit('_', 2) 
        if len(parts) == 3:
            action, author, count = parts
        else:
            action, author = "unknown", "unknown"
            
        stratify_key = f"{action}_{author}"
        
        clip_data.append({
            'filename': filename,
            'stratify_key': stratify_key
        })

    df = pd.DataFrame(clip_data)
    
    if df.empty:
        print(f"No CSV files found in {input_dir}. Did create_clips run successfully?")
        return

    # Handle Singletons (Classes with only 1 instance go to train)
    class_counts = df['stratify_key'].value_counts()
    singletons = class_counts[class_counts == 1].index
    
    df_singletons = df[df['stratify_key'].isin(singletons)]
    df_multi = df[~df['stratify_key'].isin(singletons)]
    
    # Perform the split
    if not df_multi.empty:
        train_df, test_df = train_test_split(
            df_multi, 
            test_size=test_size, 
            stratify=df_multi['stratify_key'],
            random_state=42
        )
    else:
        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        
    train_df = pd.concat([train_df, df_singletons])
    
    # Copy files
    for _, row in train_df.iterrows():
        shutil.copy(os.path.join(input_dir, row['filename']), os.path.join(train_dir, row['filename']))
        
    for _, row in test_df.iterrows():
        shutil.copy(os.path.join(input_dir, row['filename']), os.path.join(test_dir, row['filename']))
        
    print(f"--- Split Complete ---")
    print(f"Total Clips: {len(df)}")
    print(f"Training Set: {len(train_df)} clips")
    print(f"Testing Set: {len(test_df)} clips")

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Define your raw input files
    # -> UPDATE THESE to match your actual file structure
    BASE_DIR = "Data/Phase2"
    
    # Put the exact names of the folders you want to skip here
    exclude_folders = ["Train_clips","Test_clips","instance_clips","Alfred", "Kevin", "Parnika"] 
    
    raw_files = []
    
    # 2. Dynamically scan for CSV files
    if os.path.exists(BASE_DIR):
        print(f"Scanning '{BASE_DIR}' for CSV files...")
        for folder_name in os.listdir(BASE_DIR):
            folder_path = os.path.join(BASE_DIR, folder_name)
            
            # Check if it's actually a folder and not in our exclude list
            if os.path.isdir(folder_path):
                if folder_name in exclude_folders:
                    print(f"  -> Skipping excluded folder: {folder_name}")
                    continue
                
                # If approved, grab all CSVs inside
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        full_path = os.path.join(folder_path, file_name)
                        raw_files.append(full_path)
                        print(f"  -> Found: {full_path}")
    else:
        print(f"ERROR: Base directory '{BASE_DIR}' not found. Check your path.")
        
    if not raw_files:
        print("\nNo CSV files found to process. Exiting.")
        exit()
    # 2. Define your directory structure
    CLIPS_DIR = "Data/Phase2/instance_clips"
    TRAIN_DIR = "Data/Phase2/Train_clips"
    TEST_DIR  = "Data/Phase2/Test_clips"
    
    # 3. Run Step 1: Extract the padded clips
    print("=== STEP 1: EXTRACTING CLIPS ===")
    create_clips(file_paths=raw_files, output_dir=CLIPS_DIR, pad_frames=20)
    
    # 4. Run Step 2: Split into Train and Test
    print("\n=== STEP 2: SPLITTING DATA ===")
    split_clips(input_dir=CLIPS_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR, test_size=0.2)
    
    print("\nPipeline execution complete! Your data is ready.")