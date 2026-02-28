import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# MediaPipe connections
mp_connections = [
    (11,12), (12,14), (14,16), (11,13), (13,15),
    (12,24), (24,26), (26,28), (28,32), (11,23),
    (23,25), (25,27), (27,31),(23, 24)
]

# Indices removed during dataset creation
remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]


def reconstruct_points(row):
    """
    Reconstructs all 33 MediaPipe landmarks from the reduced CSV row.
    Missing ones become None.
    """
    values = row[:-1]  # exclude label
    label = row[-1]

    kept_indices = [i for i in range(33) if i not in remove_indices]
    assert len(values) == len(kept_indices) * 4

    points = {}
    ptr = 0

    # Add kept indices
    for ki in kept_indices:
        x = values[ptr]
        y = values[ptr + 1]
        z = values[ptr + 2]
        v = values[ptr + 3]
        points[ki] = (x, y, z, v)
        ptr += 4

    # Add removed indices as None
    for ri in remove_indices:
        points[ri] = None

    return points, label


def draw_skeleton(frame, points, label, W, H):
    """
    Draw white skeleton on a black frame.
    """
    # Draw bones
    for start, end in mp_connections:
        if points[start] is None or points[end] is None:
            continue

        x1, y1, _, _ = points[start]
        x2, y2, _, _ = points[end]

        # Convert normalized coords → pixels
        p1 = (int(x1 * W), int(y1 * H))
        p2 = (int(x2 * W), int(y2 * H))

        cv2.line(frame, p1, p2, (255, 255, 255), 2)

    # Draw joints
    for i, lm in points.items():
        if lm is None:
            continue
        x, y, z, v = lm
        cx, cy = int(x * W), int(y * H)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    # Put label
    cv2.putText(frame, f"{label}", (W - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return frame


def create_video(csv_path, output_video, W=640, H=480, fps=30):
    """
    Generates a black-background skeleton video from pose CSV.
    """
    df = pd.read_csv(csv_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    print(f"Creating video: {output_video}")

    for idx, row in df.iterrows():
        points, label = reconstruct_points(row.values)

        # Black background
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Draw skeleton
        frame = draw_skeleton(frame, points, label, W, H)

        writer.write(frame)

    writer.release()
    print("Done!")


if __name__ == "__main__":
    from pathlib import Path

csv_path = "Data/Train_Test_Data/Seperated_By_Person/Clips_Split_80_20/Test/lead_hook_Christian_32.csv"
name = Path(csv_path).parent.name

# 1. Define the separate parts of your output path
out_folder = "Visualized_Data"
base_name = f"{name}_output_pose_video"
ext = ".mp4"

# 2. Create the initial Path object
output_video_path = Path(f"{out_folder}/{base_name}{ext}")

# 3. If the file exists, start a counter and update the name until it's unique
counter = 1
while output_video_path.exists():
    output_video_path = Path(f"{out_folder}/{base_name}_{counter}{ext}")
    counter += 1

# 4. Convert the final unique Path back to a regular string 
# (Most video libraries like OpenCV prefer standard strings over Path objects)
output_video = str(output_video_path)

print(output_video)

create_video(csv_path, output_video, W=640, H=480, fps=30)
