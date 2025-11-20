import os
import pandas as pd
import numpy as np

# Indices removed originally (if you used them before)
remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

# Left/right pairs for MediaPipe (33 keypoints). Adjust if your numbering differs.
LEFT_RIGHT_PAIRS = [
    (11, 12), (23, 24), (25, 26), (27, 28),
    (13, 14), (15, 16), (29, 30), (31, 32)
]


def flip_dataframe(df):
    """
    Flip horizontally:
    - x --> 1 - x
    - swap left/right columns (x,y,z,v for each)
    """
    df_flipped = df.copy()

    # Flip x columns
    for col in df.columns:
        if col.startswith("x"):
            df_flipped[col] = 1.0 - df_flipped[col]

    # Swap left/right landmark columns (x,y,z,v)
    axes = ['x', 'y', 'z', 'v']
    for left, right in LEFT_RIGHT_PAIRS:
        if left in remove_indices or right in remove_indices:
            continue
        for axis in axes:
            cL = f"{axis}{left}"
            cR = f"{axis}{right}"
            if cL in df.columns and cR in df.columns:
                # swap
                tmp = df_flipped[cL].copy()
                df_flipped[cL] = df_flipped[cR]
                df_flipped[cR] = tmp

    return df_flipped

def apply_shift_one_sequence(df, x_range=0.1, y_range=0.1):
    """Apply one random uniform shift (same for all frames)."""
    dx = np.random.uniform(-x_range, x_range)
    dy = np.random.uniform(-y_range, y_range)
    df_shifted = df.copy()
    for col in df_shifted.columns:
        if col.startswith("x"):
            df_shifted[col] = df_shifted[col] + dx
        elif col.startswith("y"):
            df_shifted[col] = df_shifted[col] + dy
    return df_shifted, dx, dy

def apply_jitter(df, sigma=0.01):
    """Add per-frame small Gaussian noise to x,y columns."""
    df_jitter = df.copy()
    for col in df_jitter.columns:
        if col.startswith("x") or col.startswith("y"):
            noise = np.random.normal(0.0, sigma, size=len(df_jitter))
            df_jitter[col] = df_jitter[col] + noise
    return df_jitter

def augment_csv_make_shifts_and_flips(input_csv,
                                      output_dir,
                                      num_shifts=10,
                                      x_range=0.1,
                                      y_range=0.1,
                                      jitter_sigma=0.01,
                                      flip_prob=1.0):
    """
    Create num_shifts shifted CSVs; for each shifted CSV also save a flipped copy.
    flip_prob controls whether the flipped copy is always made (1.0) or sometimes (0.5).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    base = os.path.splitext(os.path.basename(input_csv))[0]

    for i in range(num_shifts):
        # 1) shift
        shifted, dx, dy = apply_shift_one_sequence(df, x_range=x_range, y_range=y_range)
        # 2) jitter (per-frame)
        shifted = apply_jitter(shifted, sigma=jitter_sigma)

        out_shift = os.path.join(output_dir, f"{base}_shift{i}.csv")
        shifted.to_csv(out_shift, index=False)
        print(f"Saved shifted: {out_shift} (dx={dx:.3f}, dy={dy:.3f})")

        # 4) create flipped version of the shifted file (always in this script)
        # If you want randomness you can use flip_prob < 1.0; here we always make flipped version.
        if np.random.rand() <= flip_prob:
            flipped = flip_dataframe(shifted)
            out_flip = os.path.join(output_dir, f"{base}_shift{i}_flip.csv")
            flipped.to_csv(out_flip, index=False)
            print(f"Saved flipped:  {out_flip}")

if __name__ == "__main__":
    # Example usage
    input_csv = "/Users/christianchen/CV_to_StreetFighter/Data/pose_data_kevin_labeled2.csv"
    output_dir = "DataAugmentation/AugmentedData"
    augment_csv_make_shifts_and_flips(
        input_csv=input_csv,
        output_dir=output_dir,
        num_shifts=10,        # 10 random shifts
        x_range=0.1,          # ±10% shift
        y_range=0.1,          # ±10% shift
        jitter_sigma=0.005,    # 0.5% jitter applied after shift
        flip_prob=1.0         # always create the flipped version
    )
