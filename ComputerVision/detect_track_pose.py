import cv2 as cv
import csv
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]
movement_threshold = 0.05

def record_pose_data(mode_test=False, csv_filename="pose_data.csv", movement_info=None, prediction_callback=None):
    """
    Capture pose data from webcam, save to CSV or return vectors, 
    record video (only if mode_test=False), and display movement info.
    """
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    pose_vectors = []

    if not mode_test:
        # Video writer for output
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_writer = cv.VideoWriter("pose_recording.mp4", fourcc, 20, (640, 480))

        # CSV writer
        csv_file = open(csv_filename, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        header = []
        for i in range(33 - len(remove_indices)):
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        csv_writer.writerow(header)

    # Initialize movement_info if not provided
    if movement_info is None:
        movement_info = {"direction": "", "x_diff": 0.0}
    prev_center_x = None
    prediction_text = ""

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
            frame_resized = cv.resize(frame, (640, 480))
            h, w = frame_resized.shape[:2]
            img_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

                # Draw landmarks
                #for lm in lms_filtered:
                    #cx, cy = int(lm.x * w), int(lm.y * h)
                    #cv.circle(frame_resized, (cx, cy), 5, (0, 255, 0), -1)

                # Draw connections
                #connections = [c for c in mp_pose.POSE_CONNECTIONS
                               #if c[0] not in remove_indices and c[1] not in remove_indices]
                #for start_idx, end_idx in connections:
                    #start = lms[start_idx]
                    #end = lms[end_idx]
                    #x1, y1 = int(start.x * w), int(start.y * h)
                    #x2, y2 = int(end.x * w), int(end.y * h)
                    #cv.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Pose vector
                vector = np.array([coord for lm in lms_filtered for coord in (lm.x, lm.y, lm.z, lm.visibility)], dtype=np.float32)
                
                # Get prediction from callback if provided
                if mode_test and prediction_callback:
                    prediction_text = prediction_callback(vector)

                if mode_test:
                    pose_vectors.append(vector)
                    yield vector
                else:
                    csv_writer.writerow(vector)
                    video_writer.write(frame_resized)

                # Calculate movement info (left/right)
                center_x = np.mean([lm.x for lm in lms_filtered])
                movement_info['center_x'] = center_x
                if prev_center_x is not None:
                    x_diff = center_x - prev_center_x
                    movement_info['x_diff'] = x_diff
                    if x_diff > 0.005:
                        movement_info['direction'] = "MOVE RIGHT"
                    elif x_diff < -movement_threshold:
                        movement_info['direction'] = "MOVE LEFT"
                    else:
                        movement_info['direction'] = "STATIONARY"
                prev_center_x = center_x

            else:
                movement_info['direction'] = "NO POSE DETECTED"
                movement_info['x_diff'] = 0.0
                movement_info['center_x'] = None

            # Display movement info
            direction = movement_info['direction']
            color = (0, 255, 255)
            if "RIGHT" in direction:
                color = (0, 255, 0)
            elif "LEFT" in direction or "NO POSE" in direction:
                color = (0, 0, 255)

            cv.putText(frame_resized, direction, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv.putText(frame_resized, f"X Diff: {movement_info['x_diff']:.4f}", (20, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Display prediction
            if prediction_text == "kick":
                pred_color = (255, 0, 0)
            elif prediction_text == "punch":
                pred_color = (0, 0, 255)
            else:
                pred_color = (255, 255, 255)
                
            if prediction_text:
                cv.putText(frame_resized, f"Action: {prediction_text}", (20, 120), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)

            # Draw edge indicator zones
            center_x = movement_info.get('center_x')
            if center_x is not None:
                zone_width = int(0.1 * w)  # 10% of screen width
                
                # Left edge zone
                if center_x <= 0.1:
                    # Highlight left edge zone with semi-transparent green overlay
                    overlay = frame_resized.copy()
                    cv.rectangle(overlay, (0, 0), (zone_width, h), (0, 255, 0), -1)
                    cv.addWeighted(overlay, 0.3, frame_resized, 0.7, 0, frame_resized)
                else:
                    # Dim left edge zone when not active
                    overlay = frame_resized.copy()
                    cv.rectangle(overlay, (0, 0), (zone_width, h), (50, 50, 50), -1)
                    cv.addWeighted(overlay, 0.15, frame_resized, 0.85, 0, frame_resized)
                
                # Right edge zone
                if center_x >= 0.9:
                    # Highlight right edge zone with semi-transparent green overlay
                    overlay = frame_resized.copy()
                    cv.rectangle(overlay, (w - zone_width, 0), (w, h), (0, 255, 0), -1)
                    cv.addWeighted(overlay, 0.3, frame_resized, 0.7, 0, frame_resized)
                else:
                    # Dim right edge zone when not active
                    overlay = frame_resized.copy()
                    cv.rectangle(overlay, (w - zone_width, 0), (w, h), (50, 50, 50), -1)
                    cv.addWeighted(overlay, 0.15, frame_resized, 0.85, 0, frame_resized)

            cv.imshow("Pose Recording", frame_resized)

            if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    if not mode_test:
        video_writer.release()
        csv_file.close()
    cv.destroyAllWindows()
    if mode_test:
        return


if __name__ == "__main__":
    print("Starting pose recorder... Press ESC to stop.")

    for _ in record_pose_data(mode_test=False, csv_filename="pose_data.csv"):
        pass

    print("Recording stopped. Video saved as 'pose_recording.mp4' and CSV saved as 'pose_data.csv'.")
