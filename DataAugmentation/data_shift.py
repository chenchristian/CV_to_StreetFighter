import os
import pandas as pd
import numpy as np

def shift(csv_path, output_csv_path, x_range=0.2, y_range=0.2):
    """
    Apply a uniform shift to the x, y coordinates in the pose data CSV.
    
    Parameters:
    - csv_path: Path to the input CSV file containing pose data.
    - output_csv_path: Path to save the shifted pose data CSV.
    - x_range: Range of shift for x coordinates.
    - y_range: Range of shift for y coordinates.
    """
    df = pd.read_csv(csv_path)
    
    # Identify columns for x, y coordinates
    x_cols = [col for col in df.columns if col.startswith('x')]
    y_cols = [col for col in df.columns if col.startswith('y')]

    # Generate random shifts within the specified maximums
    x_shift = np.random.uniform(-x_range, x_range)
    y_shift = np.random.uniform(-y_range, y_range)
    print(f"Applying shift: x={x_shift:.3f}, y={y_shift:.3f}")

    # Apply shifts
    df[x_cols] = df[x_cols] + x_shift
    df[y_cols] = df[y_cols] + y_shift
    
    # Save the shifted data
    df.to_csv(output_csv_path, index=False)
    
if __name__ == "__main__":
    # Example usage
    csv_file = "Data/pose_data_labeled_1.csv"
    output_file = "DataAugmentation/AugmentedData/pose_data_labeled_1_shifted.csv"
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shift(csv_file, output_file, x_range=0.2, y_range=0.2)