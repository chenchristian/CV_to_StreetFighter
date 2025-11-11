import cv2 as cv
import csv
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

def record_pose_data(mode_test=False, csv_filename="pose_data.csv", movement_info=None):
    """
    Capture pose data from webcam and either save to CSV or return vectors for testing.
    
    Args:
        mode_test (bool): True = return vector for model, False = save to CSV
        csv_filename (str): file to save data if mode_test=False
        movement_info (dict): Optional dict to share movement direction info for display
                             Should have 'direction' key that gets updated
    
    Returns:
        list of np.array if mode_test=True, else None
    """
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    pose_vectors = []  # store vectors if mode_test=True

    if not mode_test:
        csv_file = open(csv_filename, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        header = []
        for i in range(33 - len(remove_indices)):
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        csv_writer.writerow(header)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

                # Draw landmarks
                for lm in lms_filtered:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Draw connections
                connections = [c for c in mp_pose.POSE_CONNECTIONS
                               if c[0] not in remove_indices and c[1] not in remove_indices]
                for start_idx, end_idx in connections:
                    start = lms[start_idx]
                    end = lms[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Bounding box
                xs = [int(lm.x * w) for lm in lms_filtered if lm.visibility > 0.5]
                ys = [int(lm.y * h) for lm in lms_filtered if lm.visibility > 0.5]
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    pad = 25
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv.putText(frame, "Bounding Box", (x_min, y_min - 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Pose vector
                vector = []
                for lm in lms_filtered:
                    vector += [lm.x, lm.y, lm.z, lm.visibility]
                vector = np.array(vector, dtype=np.float32)

                if mode_test:
                    pose_vectors.append(vector)

                    #yielding is like returning doing it live and doesnt have to stop the program.
                    yield vector
                else:
                    csv_writer.writerow(vector)
            else:
                # No pose detected
                if movement_info is not None:
                    movement_info['direction'] = "NO POSE DETECTED"
                    movement_info['x_diff'] = 0.0
            
            # Display movement direction on screen if available
            if movement_info is not None and 'direction' in movement_info:
                direction = movement_info['direction']
                # Choose color based on direction
                if "RIGHT" in direction:
                    color = (0, 255, 0)  # Green for right
                elif "LEFT" in direction:
                    color = (0, 0, 255)  # Red for left
                elif "NO POSE" in direction:
                    color = (0, 0, 255)  # Red for no pose
                else:
                    color = (0, 255, 255)  # Yellow for stationary
                
                # Display movement direction on screen
                cv.putText(frame, direction, (50, 50),
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Display coordinate difference if available
                if 'x_diff' in movement_info:
                    x_diff = movement_info['x_diff']
                    diff_text = f"X Diff: {x_diff:.6f}"
                    cv.putText(frame, diff_text, (50, 100),
                              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv.imshow("Pose Recording", frame)
            if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    cv.destroyAllWindows()
    if not mode_test:
        csv_file.close()
    if mode_test:
        return 
    





