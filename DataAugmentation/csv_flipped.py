import pandas as pd
import os

def flip_x(data, output_path):
    df = pd.read_csv(data)
    df_flipped = df.copy()
    for i in range(len(df_flipped.columns)):
        if("x" in df_flipped.columns[i]):
            for j in range(len(df)):
                df_flipped.iloc[j, i] = round(1 - df.iloc[j, i], 7)

    output_path = os.path.join(output_path, "pose_data_labeled_flipped.csv")
    df_flipped.to_csv(output_path, index=False)
    print(f"Flipped data saved to: {output_path}")

if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(PARENT_DIR)
    path = os.path.join(BASE_DIR, "Data")
    file = "pose_data_labeled.csv"
    output_path = "AugmentedData"
    data = os.path.join(path, file)
    flip_x(data, output_path)
