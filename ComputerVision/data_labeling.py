import os
import cv2
import pandas as pd
import numpy as np

# -----------------------------
# Label Function
# -----------------------------
def label(csv_path, video_path):
    df = pd.read_csv(csv_path)
    num_frames = len(df)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    labels = ["idle"] * num_frames
    frame_idx = 0
    paused = True
    current_label = "idle"
    last_label_frame = 0

    cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # HUD
        cv2.putText(frame, f"Frame: {frame_idx+1}/{num_frames}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"Current Label: {current_label}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, "P=punch | K=kick | I=idle", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, "A=left | D=right | SPACE=play/pause | Q=quit", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Labeling Tool", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            current_label = "punch"
        elif key == ord('k'):
            current_label = "kick"
        elif key == ord('i'):
            current_label = "idle"
        elif key == 32:  # SPACE
            paused = not paused
        elif key == ord('a'):
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('d'):
            frame_idx = min(num_frames - 1, frame_idx + 1)

        # Interpolation
        start = min(last_label_frame, frame_idx)
        end = max(last_label_frame, frame_idx)
        for f in range(start, end + 1):
            labels[f] = current_label
        last_label_frame = frame_idx

        # Auto-advance if not paused
        if not paused:
            frame_idx += 1
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1
                paused = True

    # Save labeled CSV
    df["label"] = labels
    data_folder = os.path.join("Data")
    os.makedirs(data_folder, exist_ok=True)
      # Save labeled CSV with auto-incrementing name
    base_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., pose_data
    counter = 1

    # Find next available filename
    while True:
        output_name = f"{base_name}_labeled_{counter}.csv"
        output_path = os.path.join(data_folder, output_name)
        if not os.path.exists(output_path):
            break
        counter += 1

    df.to_csv(output_path, index=False)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Labeled CSV saved to: {output_path}")


# -----------------------------
# Verify Function
# -----------------------------
def verify(csv_path, video_path):
    df = pd.read_csv(csv_path)
    num_frames = len(df)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay_ms = int(1000 / fps)

    frame_idx = 0
    paused = True

    cv2.namedWindow("Verify Labels", cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        label_text = df["label"].iloc[frame_idx]
        cv2.putText(frame, f"Frame: {frame_idx+1}/{num_frames}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"Label: {label_text}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, "SPACE=pause/play | A=left | D=right | Q=quit", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Verify Labels", frame)

        key = cv2.waitKey(frame_delay_ms if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            frame_idx = max(0, frame_idx - 1)
            paused = True
        elif key == ord('d'):
            frame_idx = min(num_frames - 1, frame_idx + 1)
            paused = True

        if not paused:
            frame_idx += 1
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1
                paused = True

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    #remember to change which data you are getting label from
    csv_file = "pose_data.csv"
    csv_file_verify = "Data/pose_data_labeled_3.csv"
    video_file = "pose_recording.mp4"

    #change to label()/verify to change functionality
    verify(csv_file_verify, video_file)
