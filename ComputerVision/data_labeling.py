import os
import cv2
import pandas as pd
import numpy as np

def draw_text_outline(img, text, pos, font, scale, thickness, text_color, outline_color=(0,0,0)):
    x, y = pos

    # Outline (draw thicker black text behind)
    cv2.putText(img, text, (x, y), font, scale, outline_color, thickness + 2, cv2.LINE_AA)

    # Main text
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

# -----------------------------
# Label Function
# -----------------------------
def label(csv_path, video_path):
    import os
    import cv2
    import pandas as pd
    import numpy as np

    # -----------------------------
    # Helper: Draw Text with Outline
    # -----------------------------
    def draw_text_outline(img, text, pos, font, scale, thickness, text_color, outline_color=(0,0,0)):
        x, y = pos
        # Draw outline
        cv2.putText(img, text, (x, y), font, scale, outline_color, thickness + 2, cv2.LINE_AA)
        # Draw main text
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(csv_path)
    num_frames = len(df)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    # Key -> Move mapping
    key_map = {
        # Kicks
        ord('q'): "rear_low_kick",
        ord('w'): "side_kick",
        ord('e'): "spinning_back_high_kick",
        ord('r'): "crouching_low_sweep",
        # Punches
        ord('a'): "jab",
        ord('s'): "cross",
        ord('d'): "lead_hook",
        ord('f'): "rear_hook",
        ord('g'): "uppercut",
        ord('h'): "jumping_cross",
        # Specials
        ord('z'): "hadouken",
        ord('x'): "shoryuken",
        ord('c'): "grab",   
        # Idle
        ord('i'): "idle"
    }

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

        # -----------------------------
        # HUD
        # -----------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Frame + current label
        draw_text_outline(frame, f"Frame: {frame_idx+1}/{num_frames}",
                          (20,40), font, 1.0, 2, (255,255,255))
        draw_text_outline(frame, f"Current Label: {current_label}",
                          (20,80), font, 1.0, 2, (255,255,255))

                # Column layout
        start_y = 140
        line_gap = 25  # slightly smaller spacing
        col1_x = 20
        col2_x = 170
        col3_x = 340  # moved closer so it fits on most screens

        # Punches (orange)
        draw_text_outline(frame, "PUNCHES", (col1_x, start_y), font, 0.6, 1, (0,165,255))
        punches = ["A = Jab", "S = Cross", "D = Lead Hook", "F = Rear Hook", "G = Uppercut", "H = Jump Cross"]
        for i, text in enumerate(punches):
            draw_text_outline(frame, text, (col1_x, start_y + (i+1)*line_gap), font, 0.5, 1, (255,255,255))

        # Kicks (green)
        draw_text_outline(frame, "KICKS", (col2_x, start_y), font, 0.6, 1, (0,255,0))
        kicks = ["Q = Rear Low Kick", "W = Side Kick", "E = Spinning Back", "High Kick", "R = Low Sweep"]
        for i, text in enumerate(kicks):
            draw_text_outline(frame, text, (col2_x, start_y + (i+1)*line_gap), font, 0.5, 1, (255,255,255))

        # Specials + Idle (purple)
        draw_text_outline(frame, "SPECIAL / STATE", (col3_x, start_y), font, 0.6, 1, (255,0,255))
        specials = ["Z = Hadouken (fireball)", "X = Shoryuken (spinning uppercut)", "C = Grab", "I = Idle"]
        for i, text in enumerate(specials):
            draw_text_outline(frame, text, (col3_x, start_y + (i+1)*line_gap), font, 0.5, 1, (255,255,255))

        # Controls footer
        draw_text_outline(frame, "Comma , = Back | Period . = Forward | 0 = Quit",
                          (20, frame.shape[0] - 20), font, 0.5, 1, (200,200,200))


        # -----------------------------
        # Show frame
        # -----------------------------
        cv2.imshow("Labeling Tool", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('0'):
            break
        elif key in key_map:
            current_label = key_map[key]
        elif key == 81 or key == ord(','):  # left arrow fallback
            frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == ord('.'):  # right arrow fallback
            frame_idx = min(num_frames - 1, frame_idx + 1)

        # Interpolation
        start = min(last_label_frame, frame_idx)
        end = max(last_label_frame, frame_idx)
        for f in range(start, end + 1):
            labels[f] = current_label
        last_label_frame = frame_idx


    # -----------------------------
    # Save CSV
    # -----------------------------
    df["label"] = labels
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    data_folder = os.path.join(parent, "Data/Phase2")
    os.makedirs(data_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    counter = 1
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
    frame_delay_ms = int(1000 / fps) if fps > 0 else 33

    frame_idx = 0
    paused = True

    cv2.namedWindow("Verify Labels", cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        label_text = df["label"].iloc[frame_idx]

        cv2.putText(frame, f"Frame: {frame_idx+1}/{num_frames}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.putText(frame, f"Label: {label_text}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.putText(frame, "SPACE=play/pause | ,=back | .=forward | 0=quit",
                    (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Verify Labels", frame)

        key = cv2.waitKey(frame_delay_ms if not paused else 0) & 0xFF

        if key == ord('0'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord(','):
            frame_idx = max(0, frame_idx - 1)
            paused = True
        elif key == ord('.'):
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

    csv_file = "pose_data.csv"                     # raw pose data
    csv_file_verify = "Data/Phase1/pose_data_labeled_1.csv"   # labeled file
    video_file = "pose_recording.mp4"

    MODE = "label"   # change to "label", "verify", "" 

    if MODE == "label":
        label(csv_file, video_file)
    elif MODE == "verify":
        verify(csv_file_verify, video_file)
    

